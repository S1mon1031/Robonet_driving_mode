#!/usr/bin/env python3

###############################################################################
# 自动驾驶数据采集工具
# 功能: 从自动驾驶bag包中提取轨迹跟踪数据
# 输入: Planning、Localization的topic数据(bag文件)
# 输出: CSV文件,包含当前位姿、期望轨迹、响应轨迹
# 10s轨迹
# python3 collect_auto_driving_data.py -b *.record -o output.csv -td 10.0
###############################################################################

import argparse
import sys
import math
import csv
import glob
import os

# 设置 Python 路径
sys.path.insert(0, '/apollo')
sys.path.insert(0, '/apollo/bazel-bin')
sys.path.insert(0, '/apollo/bazel-genfiles')
sys.path.insert(0, '/apollo/cyber/python')
sys.path.insert(0, '/apollo/bazel-bin/cyber/python/internal')
sys.path.insert(0, '/apollo/bazel-bin/modules')

from cyber_py3.record import RecordReader
from modules.common_msgs.planning_msgs import planning_pb2
from modules.common_msgs.localization_msgs import localization_pb2
from modules.common_msgs.chassis_msgs import chassis_pb2
from modules.common_msgs.control_msgs import control_cmd_pb2

# 常量定义
PLANNING_TOPIC = "/apollo/planning"
LOCALIZATION_TOPIC = "/apollo/localization/pose"
CHASSIS_TOPIC = "/apollo/canbus/chassis"
CONTROL_TOPIC = "/apollo/control"

DESIRED_TRAJ_DURATION = 5.0   # 默认值，运行时由 --traj_duration 覆盖
RESPONSE_TRAJ_DURATION = 5.0  # 默认值，运行时由 --traj_duration 覆盖


def quaternion_to_euler(qx, qy, qz, qw):
    """四元数转欧拉角(roll, pitch, yaw)"""
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm == 0:
        return 0.0, 0.0, 0.0
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def lowpass_filter_v(loc_list, alpha=0.05):
    """一阶低通滤波平滑速度，alpha越小越平滑"""
    if len(loc_list) < 2:
        return
    filtered = loc_list[0]['v']
    for loc in loc_list:
        filtered = alpha * loc['v'] + (1 - alpha) * filtered
        loc['v'] = filtered


def compute_acceleration_bidirectional(loc_list):
    """双向滤波计算加速度"""
    n = len(loc_list)
    if n < 3:
        return

    MIN_DT = 0.01  # 最小时间间隔，防止dt过小导致加速度爆炸

    # 前向滤波
    forward_a = [0.0] * n
    for i in range(1, n):
        dt = loc_list[i]['timestamp'] - loc_list[i-1]['timestamp']
        # if dt >= MIN_DT:
        forward_a[i] = (loc_list[i]['v'] - loc_list[i-1]['v']) / 0.1

    # 后向滤波
    backward_a = [0.0] * n
    for i in range(n-2, -1, -1):
        dt = loc_list[i+1]['timestamp'] - loc_list[i]['timestamp']
        # if dt >= MIN_DT:
        backward_a[i] = (loc_list[i+1]['v'] - loc_list[i]['v']) / 0.1

    # 平均
    for i in range(n):
        loc_list[i]['a'] = (forward_a[i] + backward_a[i]) / 2.0


def lowpass_filter_a(loc_list, alpha=0.1):
    """一阶低通滤波平滑加速度"""
    if len(loc_list) < 2:
        return
    filtered = loc_list[0]['a']
    for loc in loc_list:
        filtered = alpha * loc['a'] + (1 - alpha) * filtered
        loc['a'] = filtered


def compute_kappa(loc_list):
    """计算曲率 kappa = angular_velocity_z / v"""
    for loc in loc_list:
        if abs(loc['v']) > 0.1:
            loc['kappa'] = loc['angular_velocity_z'] / loc['v']
        else:
            loc['kappa'] = 0.0


def read_bag_data(bag_path, traj_duration=5.0):
    """读取bag文件数据"""
    planning_list = []
    localization_list = []
    chassis_list = []
    control_list = []

    print(f"读取bag文件: {bag_path}")
    record = RecordReader(bag_path)

    for channel_name, msg, datatype, timestamp in record.read_messages():
        if channel_name == LOCALIZATION_TOPIC:
            loc_msg = localization_pb2.LocalizationEstimate()
            loc_msg.ParseFromString(msg)

            ts = loc_msg.header.timestamp_sec
            qx = loc_msg.pose.orientation.qx
            qy = loc_msg.pose.orientation.qy
            qz = loc_msg.pose.orientation.qz
            qw = loc_msg.pose.orientation.qw
            _, pitch, _ = quaternion_to_euler(qx, qy, qz, qw)

            vx = loc_msg.pose.linear_velocity.x
            vy = loc_msg.pose.linear_velocity.y
            v = vx * math.cos(loc_msg.pose.heading) + vy * math.sin(loc_msg.pose.heading)

            localization_list.append({
                'timestamp': ts,
                'pos_x': loc_msg.pose.position.x,
                'pos_y': loc_msg.pose.position.y,
                'heading': loc_msg.pose.heading,
                'pitch': pitch,
                'v': v,
                'angular_velocity_z': loc_msg.pose.angular_velocity.z,
                'a': 0.0,
                'kappa': 0.0,
            })

        elif channel_name == CHASSIS_TOPIC:
            chassis_msg = chassis_pb2.Chassis()
            chassis_msg.ParseFromString(msg)

            chassis_list.append({
                'timestamp': chassis_msg.header.timestamp_sec,
                'throttle_percentage': chassis_msg.throttle_percentage,
                'brake_percentage': chassis_msg.brake_percentage,
                'steering_angle': chassis_msg.steering_angle,
                'gear_location': (0 if chassis_msg.gear_location == 0
                                  else -1 if chassis_msg.gear_location == 2
                                  else chassis_msg.forward_gear_level),  # D档取forward_gear_level
                'oil_consumption': chassis_msg.oil_consumption,
                'mileage': chassis_msg.mileage,
                'engine_torque': chassis_msg.engine_torque,
            })

        elif channel_name == CONTROL_TOPIC:
            control_msg = control_cmd_pb2.ControlCommand()
            control_msg.ParseFromString(msg)

            # 处理 control_backup_field_a: 2.2->1(重载), 1.2->0(轻载)
            load_value = 0
            if hasattr(control_msg, 'control_backup_field_a'):
                field_a = control_msg.control_backup_field_a
                if abs(field_a - 2.2) < 0.1:
                    load_value = 1
                elif abs(field_a - 1.2) < 0.1:
                    load_value = 0

            control_list.append({
                'timestamp': control_msg.header.timestamp_sec,
                'control_throttle': control_msg.throttle,
                'control_brake': control_msg.brake,
                'load': load_value,
            })

        elif channel_name == PLANNING_TOPIC:
            plan_msg = planning_pb2.ADCTrajectory()
            plan_msg.ParseFromString(msg)

            ts = plan_msg.header.timestamp_sec
            traj_points = []

            for tp in plan_msg.trajectory_point:
                if tp.relative_time <= traj_duration + 1.0:  # 多读1s buffer
                    traj_points.append({
                        'x': tp.path_point.x,
                        'y': tp.path_point.y,
                        's': tp.path_point.s,
                        'v': tp.v,
                        'a': tp.a,
                        'kappa': tp.path_point.kappa,
                        'relative_time': tp.relative_time,
                    })

            if traj_points:
                planning_list.append({
                    'timestamp': ts,
                    'trajectory': traj_points,
                })

    localization_list.sort(key=lambda x: x['timestamp'])
    planning_list.sort(key=lambda x: x['timestamp'])
    chassis_list.sort(key=lambda x: x['timestamp'])
    control_list.sort(key=lambda x: x['timestamp'])

    print(f"读取完成: localization {len(localization_list)} 条, planning {len(planning_list)} 条, chassis {len(chassis_list)} 条, control {len(control_list)} 条")
    return localization_list, planning_list, chassis_list, control_list


def find_closest_index(data_list, target_ts):
    """二分查找最接近的时间戳索引"""
    if not data_list:
        return -1

    left, right = 0, len(data_list) - 1
    closest_idx = 0
    min_diff = abs(data_list[0]['timestamp'] - target_ts)

    while left <= right:
        mid = (left + right) // 2
        diff = abs(data_list[mid]['timestamp'] - target_ts)

        if diff < min_diff:
            min_diff = diff
            closest_idx = mid

        if data_list[mid]['timestamp'] < target_ts:
            left = mid + 1
        else:
            right = mid - 1

    return closest_idx


def get_response_trajectory(loc_list, start_idx, start_ts, traj_duration=5.0):
    """获取响应轨迹（实际车辆位姿），从当前时刻开始取 traj_duration 秒，重采样到0.1s间隔"""
    n_points = int(round(traj_duration / 0.1))  # 5s→50, 10s→100
    raw = []

    for i in range(start_idx, len(loc_list)):
        loc = loc_list[i]
        dt = loc['timestamp'] - start_ts
        if dt < 0:
            continue
        if dt > traj_duration + 0.2:
            break
        raw.append({
            'x': loc['pos_x'],
            'y': loc['pos_y'],
            'v': loc['v'],
            'a': loc['a'],
            'kappa': loc['kappa'],
            'relative_time': dt,
        })

    if not raw:
        return []

    # 重采样到 0.1s 间隔
    target_times = [round((i + 1) * 0.1, 1) for i in range(n_points)]
    response_traj = []
    for t in target_times:
        # 找最近的原始点
        best = min(raw, key=lambda p: abs(p['relative_time'] - t))
        response_traj.append({
            'x': best['x'],
            'y': best['y'],
            'v': best['v'],
            'a': best['a'],
            'kappa': best['kappa'],
            'relative_time': t,
        })

    return response_traj


def match_desired_trajectory_to_vehicle(desired_traj, vehicle_x, vehicle_y, traj_duration=5.0):
    """将期望轨迹与自车位置匹配，找到最近的起点（仅考虑relative_time>=0的点）"""
    if not desired_traj:
        return []

    # 过滤出 relative_time >= 0 的点
    future_traj = [tp for tp in desired_traj if tp['relative_time'] >= 0]

    if not future_traj:
        return []

    # 找到距离自车最近的轨迹点
    min_dist = float('inf')
    start_idx = 0

    for i, tp in enumerate(future_traj):
        dist = math.sqrt((tp['x'] - vehicle_x)**2 + (tp['y'] - vehicle_y)**2)
        if dist < min_dist:
            min_dist = dist
            start_idx = i

    # 从最近点开始截取轨迹
    matched_traj = future_traj[start_idx:]

    # 如果轨迹不足，用最后一个点补充
    if matched_traj and matched_traj[-1]['relative_time'] < traj_duration:
        last_point = matched_traj[-1].copy()
        while last_point['relative_time'] < traj_duration:
            last_point = last_point.copy()
            last_point['relative_time'] += 0.1
            matched_traj.append(last_point)

    return matched_traj


def process_data(loc_list, plan_list, chassis_list, control_list, writer, traj_duration=5.0):
    """处理数据并写入CSV"""
    total_count = 0

    for plan_idx, plan in enumerate(plan_list):
        plan_ts = plan['timestamp']
        desired_traj = plan['trajectory']

        if not desired_traj:
            continue

        # 找到最接近的localization
        loc_idx = find_closest_index(loc_list, plan_ts)
        if loc_idx < 0:
            continue

        loc = loc_list[loc_idx]
        time_diff = abs(loc['timestamp'] - plan_ts)

        if time_diff > 0.1:  # 超过100ms认为不匹配（planning周期200ms的一半）
            continue

        # 找到最接近的chassis
        chassis_idx = find_closest_index(chassis_list, plan_ts)
        chassis = chassis_list[chassis_idx] if chassis_idx >= 0 else None

        # 找到最接近的control
        control_idx = find_closest_index(control_list, plan_ts)
        control = control_list[control_idx] if control_idx >= 0 else None

        # 匹配期望轨迹到自车位置
        matched_desired_traj = match_desired_trajectory_to_vehicle(
            desired_traj, loc['pos_x'], loc['pos_y'], traj_duration)

        if not matched_desired_traj:
            continue

        # 截断到固定点数
        n_traj_points = int(round(traj_duration / 0.1))
        matched_desired_traj = matched_desired_traj[:n_traj_points]

        # 获取响应轨迹
        response_traj = get_response_trajectory(loc_list, loc_idx, plan_ts, traj_duration)

        if len(response_traj) < 5:  # 响应轨迹点太少
            continue

        # 构建CSV行
        row = [
            plan_ts,
            loc['pos_x'],
            loc['pos_y'],
            loc['heading'],
            loc['pitch'],
            loc['v'],
            loc['a'],
        ]

        # chassis信息
        if chassis:
            row.extend([
                chassis['throttle_percentage'],
                chassis['brake_percentage'],
                chassis['steering_angle'],
                chassis['gear_location'],
                chassis['oil_consumption'],
                chassis['mileage'],
                chassis['engine_torque'],
            ])
        else:
            row.extend([0.0, 0.0, 0.0, 0, 0.0, 0.0, 0])

        # control信息
        if control:
            row.extend([
                control['control_throttle'],
                control['control_brake'],
                control['load'],
            ])
        else:
            row.extend([0.0, 0.0, 0])

        # 期望轨迹（匹配后的轨迹）
        desired_data = []
        for tp in matched_desired_traj:
            desired_data.extend([tp['x'], tp['y'], tp['s'], tp['v'], tp['a'], tp['kappa']])

        # 响应轨迹（实际位姿）
        response_data = []
        for rp in response_traj:
            response_data.extend([rp['x'], rp['y'], rp['v'], rp['a'], rp['kappa']])

        row.extend(desired_data)
        row.extend(response_data)

        writer.writerow(row)
        total_count += 1

        if total_count % 100 == 0:
            print(f"已处理 {total_count} 条数据...")

    return total_count


def main():
    parser = argparse.ArgumentParser(description='自动驾驶数据采集工具')
    parser.add_argument('-b', '--bags', required=True, nargs='+', help='bag文件路径（支持shell通配符展开）')
    parser.add_argument('-o', '--output', required=True, help='输出CSV文件路径')
    parser.add_argument('-td', '--traj_duration', type=float, default=5.0,
                        help='轨迹时长（秒），支持2.0/5.0/10.0，默认5.0')
    args = parser.parse_args()

    traj_duration = args.traj_duration

    # 获取所有bag文件（支持shell展开的多个路径）
    bag_files = sorted(args.bags)
    if not bag_files:
        print(f"错误: 未找到bag文件: {args.bags}")
        return

    print(f"找到 {len(bag_files)} 个bag文件，轨迹时长: {traj_duration}s")

    # 读取所有bag
    all_loc = []
    all_plan = []
    all_chassis = []
    all_control = []

    for bag_file in bag_files:
        loc, plan, chassis, control = read_bag_data(bag_file, traj_duration)
        all_loc.extend(loc)
        all_plan.extend(plan)
        all_chassis.extend(chassis)
        all_control.extend(control)

    print(f"\n合并数据: localization {len(all_loc)} 条, planning {len(all_plan)} 条, chassis {len(all_chassis)} 条, control {len(all_control)} 条")

    all_loc.sort(key=lambda x: x['timestamp'])
    all_plan.sort(key=lambda x: x['timestamp'])
    all_chassis.sort(key=lambda x: x['timestamp'])
    all_control.sort(key=lambda x: x['timestamp'])

    # 计算加速度和曲率
    print("计算加速度和曲率...")
    lowpass_filter_v(all_loc)
    compute_acceleration_bidirectional(all_loc)
    lowpass_filter_a(all_loc)
    compute_kappa(all_loc)

    # 写入CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # CSV表头（每点字段单独命名）
        n_traj_points = int(round(traj_duration / 0.1))
        header = [
            'timestamp',
            'current_x', 'current_y', 'current_heading', 'current_pitch',
            'current_v', 'current_a',
            'throttle_percentage', 'brake_percentage', 'steering_angle',
            'gear_location', 'oil_consumption', 'mileage', 'engine_torque',
            'control_throttle', 'control_brake', 'load',
        ]
        for i in range(n_traj_points):
            header += [f'des_x_{i}', f'des_y_{i}', f'des_s_{i}',
                       f'des_v_{i}', f'des_a_{i}', f'des_kappa_{i}']
        for i in range(n_traj_points):
            header += [f'resp_x_{i}', f'resp_y_{i}', f'resp_v_{i}',
                       f'resp_a_{i}', f'resp_kappa_{i}']
        writer.writerow(header)

        total = process_data(all_loc, all_plan, all_chassis, all_control, writer, traj_duration)

    print(f"\n完成！共写入 {total} 条数据 -> {args.output}")


if __name__ == '__main__':
    main()
