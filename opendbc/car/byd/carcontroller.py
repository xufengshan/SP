import numpy as np
import math
import time
from opendbc.can.packer import CANPacker
from opendbc.car import Bus, apply_driver_steer_torque_limits, structs
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.byd import bydcan
from opendbc.car.byd.values import CarControllerParams

VisualAlert = structs.CarControl.HUDControl.VisualAlert
ButtonType = structs.CarState.ButtonEvent.Type
LongCtrlState = structs.CarControl.Actuators.LongControlState

class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)

    self.packer = CANPacker(dbc_names[Bus.pt])
    from cereal import messaging
    self.sm = messaging.SubMaster(['radarState', 'modelV2', 'longitudinalPlan'])
    self.frame = 0
    self.last_steer_frame = 0
    self.last_acc_frame = 0

    self.apply_torque_last = 0

    self.mpc_lkas_counter = 0
    self.mpc_acc_counter = 0
    self.eps_fake318_counter = 0

    self.lkas_req_prepare = 0
    self.lkas_active = 0
    self.lat_safeoff = 0

    self.steer_softstart_limit = 0
    self.steerRateLimActive = False
    self.steerRateLim = 1.0

    self.first_start = True
    self.rfss = 0 # resume from stand still
    self.sss = 0 #stand still state

    self.apply_accel_last = 0

  def update(self, CC, CS, now_nanos):
    can_sends = []

    # ?????? - ??????
    if (self.frame - self.last_steer_frame) >= CarControllerParams.STEER_STEP:
      if self.first_start:
        self.mpc_lkas_counter = int(CS.acc_mpc_state_counter + 1) & 0xF
        self.mpc_acc_counter = int(CS.acc_cmd_counter + 1) & 0xF
        self.eps_fake318_counter = int(CS.eps_state_counter + 1) & 0xF
        self.first_start = False

      apply_torque = 0

      if CC.latActive:
        if self.lkas_active:
          steer_desire = CC.actuators.torque

          if CarControllerParams.USE_STEERING_SPEED_LIMITER:
            rate_limit = np.interp(CS.out.aEgo, [8.3, 27.8], [132, 64])
            delta_rate = CS.steeringRateDegAbs - rate_limit

            if delta_rate < 0:
              self.steerRateLim -= 0.005 * delta_rate
              if delta_rate < -0.05:
                self.steerRateLimActive = False
              if self.steerRateLim > 1.0:
                self.steerRateLim = 1.0
                self.steerRateLimActive = False
            else:
              if self.steerRateLimActive:
                self.steerRateLim -= 0.005 * delta_rate
              else:
                self.steerRateLim = steer_desire
                self.steerRateLimActive = True
              if self.steerRateLim < 0:
                self.steerRateLim = 0

            new_steer_pu = np.clip(steer_desire, -self.steerRateLim, self.steerRateLim)
          else:
            new_steer_pu = steer_desire

          new_steer = int(round(new_steer_pu * CarControllerParams.STEER_MAX))

          if self.steer_softstart_limit < CarControllerParams.STEER_MAX:
            self.steer_softstart_limit = self.steer_softstart_limit + CarControllerParams.STEER_SOFTSTART_STEP
            new_steer = np.clip(new_steer, -self.steer_softstart_limit, self.steer_softstart_limit)

          apply_torque = apply_driver_steer_torque_limits(new_steer, self.apply_torque_last,
                                                          CS.out.steeringTorque, CarControllerParams)
        else:
          if CS.lkas_prepared:
            self.lkas_active = 1.0
            self.steerRateLimActive = False
            self.steerRateLim = 1.0
            self.lkas_req_prepare = 0
            self.steer_softstart_limit = 0
            self.lat_safeoff = 1
          else:
            self.lkas_req_prepare = 1

      elif self.lat_safeoff:
        if self.apply_torque_last == 0:
          self.lat_safeoff = 0
        apply_torque = apply_driver_steer_torque_limits(0, self.apply_torque_last,
                                                          CS.out.steeringTorque, CarControllerParams)
      else:
        self.lkas_req_prepare = 0
        self.steerRateLimActive = False
        self.steerRateLim = 1.0
        self.lkas_active = 0
        self.steer_softstart_limit = 0

      self.apply_torque_last = apply_torque

      self.mpc_lkas_counter = int(self.mpc_lkas_counter + 1) & 0xF
      self.eps_fake318_counter = int(self.eps_fake318_counter + 1) & 0xF
      self.last_steer_frame = self.frame

      can_sends.append(bydcan.create_steering_control(self.packer, self.CP, CS.cam_lkas,
          self.apply_torque_last, self.lkas_req_prepare, self.lkas_active, CC.hudControl, self.mpc_lkas_counter))

      can_sends.append(bydcan.create_fake_318(self.packer, self.CP, CS.esc_eps,
                                              CS.mpc_laks_output, CS.mpc_laks_reqprepare, CS.esp_lkas_CruiseActivated,
                                              True, self.eps_fake318_counter))

    # ?????? - ??MPC??,??????
    if (self.frame + 1 - self.last_acc_frame) >= CarControllerParams.ACC_STEP:
      # ??????
      self.sm.update(0)

      mpc_target_accel = CC.actuators.accel

      if CC.longActive:
        stopping = CC.actuators.longControlState == LongCtrlState.stopping
        starting = CC.actuators.longControlState == LongCtrlState.starting
        running = CC.actuators.longControlState == LongCtrlState.pid

        # ????????????(???????)
        lead_distance = getattr(CS, 'mrr_leading_dist', 199)
        v_ego = CS.out.vEgo

        # ????????????
        lead_speed = 0.0
        relative_speed = 0.0
        fusion_distance = 199
        data_source = "no_radar"

        if hasattr(self, 'sm') and self.sm.alive['radarState']:
            lead_one = self.sm['radarState'].leadOne
            if lead_one.status:
                lead_speed = lead_one.vLead if not math.isnan(lead_one.vLead) else 0.0
                relative_speed = lead_one.vRel if not math.isnan(lead_one.vRel) else 0.0
                fusion_distance = lead_one.dRel
                data_source = "radar"
            else:
                data_source = "no_lead"

        # ??????????????
        # ??MPC???,????????????
        if mpc_target_accel < 0:
            # ?????????????
            if fusion_distance < 199:
                # ????:??????????
                if relative_speed < -2.0 and fusion_distance < v_ego * 1.5:
                    # ?????,??????
                    distance_factor = 1.0  # ?????
                    speed_factor = 1.2     # ????
                else:
                    # ???????
                    distance_factor = np.interp(fusion_distance, [5.0, 30.0], [0.8, 0.4])
                    if relative_speed < -1.0:
                        speed_factor = 1.0
                    elif relative_speed < 0:
                        speed_factor = 0.7
                    else:
                        speed_factor = 0.5

                # ????:??????(????),??????
                if relative_speed < -1.0:  # ??????
                    speed_factor = 1.0
                elif relative_speed < 0:   # ??????
                    speed_factor = 0.7
                else:                      # ?????????
                    speed_factor = 0.5

                # ??????
                brake_scale = distance_factor * speed_factor
                brake_scale = np.clip(brake_scale, 0.3, 0.8)
            else:
                # ??????????
                brake_scale = 0.3

            scaled_accel = mpc_target_accel * brake_scale
        else:
            # ????????
            scaled_accel = mpc_target_accel

        # ???? - ???????
        if hasattr(self, 'last_final_accel'):
            # ??MPC?????
            if hasattr(self, 'last_mpc_accel'):
                mpc_change = abs(mpc_target_accel - self.last_mpc_accel)
                accel_change_limit = 0.1 if mpc_change > 2.0 else 0.2
            else:
                accel_change_limit = 0.25

            accel_diff = scaled_accel - self.last_final_accel
            if abs(accel_diff) > accel_change_limit:
                scaled_accel = self.last_final_accel + np.sign(accel_diff) * accel_change_limit

        self.last_mpc_accel = mpc_target_accel
        final_accel = np.clip(scaled_accel, -2.5, 1.0)  # ????????
        self.last_final_accel = final_accel

        # ??????
        if stopping and final_accel < -0.1:
          self.rfss = 0
          self.sss = CS.out.standstill
        elif starting and final_accel > 0.1 and CS.out.vEgo < 0.8:
          self.rfss = CS.out.standstill
          self.sss = 0
        elif running:
          self.rfss = 0
          self.sss = 0
      else:
        final_accel = 0
        scaled_accel = 0
        lead_speed = 0.0
        relative_speed = 0.0
        lead_distance = 199
        fusion_distance = 199
        data_source = "no_lead"
        self.sss = 0
        self.rfss = 0

      self.mpc_acc_counter = int(self.mpc_acc_counter + 1) & 0xF

      # ??????
      can_sends.append(bydcan.acc_cmd(self.packer, self.CP, CS.cam_acc,
                                     getattr(CS, 'mrr_leading_dist', 199),
                                     final_accel, self.rfss, self.sss, CC.longActive,))

      self.apply_accel_last = final_accel
      self.last_acc_frame = self.frame + 1

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / CarControllerParams.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last
    new_actuators.accel = float(self.apply_accel_last)
    new_actuators.steeringAngleDeg = float(CS.out.steeringAngleDeg)

    self.frame += 1
    return new_actuators, can_sends
