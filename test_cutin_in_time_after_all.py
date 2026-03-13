import carla
import time
import torch
import keyboard  # 用于捕获键盘输入
import cv2
import re
import math
import numpy as np
import random
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from TKG.AutoVehicle_KG_Function_3D_simple_version_V2 import AVKG_3D_simple
from TKG.AutoVehicle_EKG_Function_simple_version_V2 import AVKG_3D_simple_EKG
from nets.TKGCN_V9 import  TKGCN_V9

# 定义图像的宽度和高度
image_width = 1600
image_height = 1200


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')    # rgx 是一个正则表达式对象，用于将驼峰式命名的字符串分割为单词。
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))      # 定义一个匿名函数 name，它将字符串 x 中根据正则表达式匹配出的部分组合成用空格分隔的单词。
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]   # 使用 dir() 列出 carla.WeatherParameters 中所有以大写字母开头的属性，保存在 presets 列表中。
    # print([(getattr(carla.WeatherParameters, x), name(x)) for x in presets])
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]   # 返回一个列表，列表的每一项是 (属性值, 属性名) 的元组，属性名通过正则表达式分割后变得更易读。


def calculate_angle_between_vectors(vector_a, vector_b):
    """计算两个向量之间的夹角（以度数表示），返回值在 -180 到 180 度之间。"""
    dot_product = vector_a.x * vector_b.x + vector_a.y * vector_b.y
    magnitude_a = math.sqrt(vector_a.x ** 2 + vector_a.y ** 2)
    magnitude_b = math.sqrt(vector_b.x ** 2 + vector_b.y ** 2)
    if magnitude_a * magnitude_b == 0:
        return 0
    angle = math.acos(dot_product / (magnitude_a * magnitude_b))  # 结果以弧度为单位
    angle_degrees = math.degrees(angle)

    # 确定方向（根据叉乘的z分量）
    cross_product_z = vector_a.x * vector_b.y - vector_a.y * vector_b.x
    if cross_product_z < 0:
        angle_degrees = -angle_degrees

    return angle_degrees

def process_image(image):
    """处理CARLA采集的图像，并转换为OpenCV格式。"""
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_data = image_data.reshape((image.height, image.width, 4))  # BGRA格式
    return image_data[:, :, :3]  # 只保留BGR通道

def resize_image(image, width, height):
    """调整图像大小。"""
    return cv2.resize(image, (width, height))



def main():
    # 连接到CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)    # 设置超时时间

    device = torch.device('cuda')
    print(f'Using device: {device}')
    tkgcn = TKGCN_V9(nhid=128,pooling_rate=0.001,dropout_rate=0.01,num_classes=2,num_features=60,num_relations=75).to(device)

    # 加载预训练权重
    weight_path = 'nets/nets_weights/best_tkgcn_model_V9.pth'
    tkgcn.load_state_dict(torch.load(weight_path, map_location='cpu'))
    print('Weights are loaded!')


    ################################ random value ##############################
    # 生成一个 0 到 10 之间的随机整数
    random_int_car1 = random.randint(0, 30)
    random_int_car1_type = random.randint(0, 9)
    # print(random_int_car1)

    random_int_car2 = random.randint(0, 30)
    random_int_car2_type = random.randint(0, 9)
    # print(random_int_car2)

    random_int_front_car = random.randint(5, 8)

    random_int_weather = random.randint(1, 10)


    ####################### 7 layer list ################################
    road_structure_list = (
        "straight", ("road_region", "road_curve", "road_direction")
    )
    #######
    road_facility_list = (
        # lane
        (("lane_1", ("l_1_type", "l_1_direction", "l_1_puepose", "l_1_width")),
         ("lane_2", "lane_1", ("l_2_type", "l_2_direction", "l_2_puepose", "l_2_width")),
         ("lane_3", "lane_2", ("l_3_type", "l_3_direction", "l_3_puepose", "l_3_width")),
         ("lane_4", "lane_3", ("l_4_type", "l_4_direction", "l_4_puepose", "l_4_width")),),
        # "marker"
        (("marker_1", ("m_1_type", "m_1_color", "m_1_quality")),
         ("marker_2", ("m_2_type", "m_2_color", "m_2_quality")),),
        # "sign"
        (("sign_1", "s_1_type"),
         ("sign_2", "s_2_type"),),
        # "light"
        ("light-s", "light-state", "red"),
        # "barrier"
        ("barrier_1",
         "barrier_2",)
    )
    ########
    road_change_list = ()

    ######
    digital_information_list = (
        "communication_quality",
    )


    try:
        # 加载Town04地图
        world = client.load_world('Town04')

        # 天气   preset是一个列表，其中第一个元素是设置信息，第二个元素是名称
        preset = find_weather_presets()
        print('Carla has below kinds of weather: ')
        print(preset)
        print('Current weather is: ',preset[1][1])
        world.set_weather(preset[1][0])

        ##########   天气的内容
        weather_list = (
            "weather-attr", preset[1][1], ("w_direction", "w_level")
        )

        #########  交通参与者列表
        traffic_participants_list =()

        ego_information_list = ()

        ############ 生成自车 #########
        # 获取车辆蓝图
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[9]
        # 选择一个出生点
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[300] if len(spawn_points) > 300 else spawn_points[0]
        # 生成车辆
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        # 创建摄像头蓝图
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_width))
        camera_bp.set_attribute('image_size_y', str(image_height))
        camera_bp.set_attribute('fov', '90')
        # 添加顶置俯视摄像头
        top_view_transform = carla.Transform(carla.Location(x=2.0,z=1.8, y=0.0), carla.Rotation(pitch=0))
        top_view_camera = world.spawn_actor(camera_bp, top_view_transform, attach_to=ego_vehicle)

        # 创建俯视图像
        top_view_image = None
        def top_view_callback(image):
            nonlocal top_view_image
            top_view_image = process_image(image)
        top_view_camera.listen(top_view_callback)


        ######################### 创建前车 ##########################
        front_vehicle_bp = blueprint_library.filter('vehicle.*')[9]
        spawn_point_front = spawn_points[301] if len(spawn_points) > 300 else spawn_points[0]
        # 计算新的位置，前方移动10米
        forward_vector = spawn_point_front.get_forward_vector()  # 获取生成点的前方向量
        new_location = spawn_point_front.location + forward_vector * (15.0+random_int_front_car)  # 向前移动80米
        spawn_point_front.location = new_location  # 更新生成点位置
        # 生成前方的车辆
        front_vehicle = world.spawn_actor(front_vehicle_bp, spawn_point_front)

        ######################### 创建车流 ##########################
        # # other-1
        other_vehicle_bp = blueprint_library.filter('vehicle.*')[random_int_car1_type]
        spawn_point_other = spawn_points[298] if len(spawn_points) > 300 else spawn_points[0]
        # 计算新的位置，前方移动10米
        forward_distence = spawn_point_other.get_forward_vector()  # 获取生成点的前方向量
        new_location_other = spawn_point_other.location + forward_distence * random_int_car1 # 向前移动80米
        spawn_point_other.location = new_location_other  # 更新生成点位置
        # 生成前方的车辆
        other_vehicle = world.spawn_actor(other_vehicle_bp, spawn_point_other)


        # # other-2
        other_vehicle_bp_2 = blueprint_library.filter('vehicle.*')[0]
        spawn_point_other_2 = spawn_points[299] if len(spawn_points) > 300 else spawn_points[0]
        # 计算新的位置，前方移动10米
        forward_distence_2 = spawn_point_other_2.get_forward_vector()  # 获取生成点的前方向量
        new_location_other_2 = spawn_point_other_2.location + forward_distence_2 * random_int_car2  # 向前移动80米
        spawn_point_other_2.location = new_location_other_2  # 更新生成点位置
        # 生成前方的车辆
        other_vehicle_2 = world.spawn_actor(other_vehicle_bp_2, spawn_point_other_2)

        # # # other-3
        # other_vehicle_bp_3 = blueprint_library.filter('vehicle.*')[0]
        # spawn_point_other_3 = spawn_points[300] if len(spawn_points) > 300 else spawn_points[0]
        # # 计算新的位置，前方移动10米
        # forward_distence_3 = spawn_point_other_3.get_forward_vector()  # 获取生成点的前方向量
        # new_location_other_3 = spawn_point_other_3.location + forward_distence_3 *15.0  # 向前移动80米
        # spawn_point_other_3.location = new_location_other_3  # 更新生成点位置
        # # 生成前方的车辆
        # other_vehicle_3 = world.spawn_actor(other_vehicle_bp_3, spawn_point_other_3)

        # other-4
        # other_vehicle_bp_4 = blueprint_library.filter('vehicle.*')[2]
        # spawn_point_other_4 = spawn_points[301] if len(spawn_points) > 300 else spawn_points[0]
        # # 计算新的位置，前方移动10米
        # forward_distence_4 = spawn_point_other_4.get_forward_vector()  # 获取生成点的前方向量
        # new_location_other_4 = spawn_point_other_4.location + forward_distence_4 * 4.0  # 向前移动80米
        # spawn_point_other_4.location = new_location_other_4  # 更新生成点位置
        # # 生成前方的车辆
        # other_vehicle_4 = world.spawn_actor(other_vehicle_bp_4, spawn_point_other_4)


        ######################## 生成行人 ############################
        walker_bp = blueprint_library.filter('walker.pedestrian.*')[0]  # 选择一个行人蓝图
        walker_spawn_point = carla.Transform(
            carla.Location(
                x=new_location.x + forward_vector.x * 120.0,
                y=new_location.y + forward_vector.x * 2.2 ,
                z=new_location.z - forward_vector.z * 0.2
            ),carla.Rotation(yaw=90)
        )
        # 生成行人
        walker = world.try_spawn_actor(walker_bp, walker_spawn_point)

        # 启动自动驾驶功能
        ego_control = carla.VehicleControl()
        ego_control.throttle = 0.93  # 设置初始油门值, 0 到 1 之间
        ego_vehicle.apply_control(ego_control)
        # ego_vehicle.set_autopilot(True)

        front_control = carla.VehicleControl()
        front_control.throttle = 0.94  # 设置初始油门值, 0 到 1 之间
        front_vehicle.apply_control(front_control)

        other_vehicle_control = carla.VehicleControl()
        other_vehicle_control.throttle = 0.74  # 设置初始油门值, 0 到 1 之间
        other_vehicle.apply_control(other_vehicle_control)
        #
        other_vehicle_control2 = carla.VehicleControl()
        other_vehicle_control2.throttle = 0.74  # 设置初始油门值, 0 到 1 之间
        other_vehicle_2.apply_control(other_vehicle_control2)
        #
        # other_vehicle_control3 = carla.VehicleControl()
        # other_vehicle_control3.throttle = 0.75 # 设置初始油门值, 0 到 1 之间
        # other_vehicle_3.apply_control(other_vehicle_control3)
        #
        # other_vehicle_control4 = carla.VehicleControl()
        # other_vehicle_control4.throttle = 0.7  # 设置初始油门值, 0 到 1 之间
        # other_vehicle_4.apply_control(other_vehicle_control4)

        print("在主车右前方 20 米生成了一个自动驾驶车辆。")
        print("车辆已启动自动驾驶。按 'Esc' 键退出程序。")

        # 让行人向y方向移动
        print("行人开始移动...")
        walker_speed = 0.1 # 行人移动速度 (米/秒)
        start_time = time.time()

        frame_total = 0

        scene_final_list = []    # 用于后续的计算
        pic_list = []
        event_start_frame = 0
        if_start = False

        event_start_cutin = 0
        if_start_cutin = False

        try:
            vehicles = world.get_actors().filter('vehicle.*')
            pedestrians = world.get_actors().filter('walker.pedestrian.*')

            # 用于存放行为的状态
            behavior_state_name = []
            behavior_state_name_speed = []

            scene_final=()
            scene_length_final =10
            space_frame = 10

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_fps = 60
            video_save_path = 'result/test_cut_in_1-19.mp4'
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, (800, 600))

            final_img = np.ones((600, 800, 3), dtype=np.uint8) * 255
            final_img_all = np.ones((1200, 1600, 3), dtype=np.uint8) * 255

            # 主循环
            while True:
                print('\n')
                print('This is frame : ',frame_total)
                frame_total += 1

                # 实时计算他车与自车之间的距离
                if walker and front_vehicle:
                    ego_forward_vector = ego_vehicle.get_transform().get_forward_vector()  # 自车朝向向量
                    front_location = front_vehicle.get_location()
                    front_vehicle_bounding_box = front_vehicle.bounding_box  # 获取车辆的长度
                    walker_location = walker.get_location()
                    ego_location = ego_vehicle.get_location()
                    distance_ego = math.sqrt(
                        (front_location.x - ego_location.x) ** 2 +
                        (front_location.y - ego_location.y) ** 2 +
                        (front_location.z - ego_location.z) ** 2
                    ) - front_vehicle_bounding_box.extent.x * 2
                    # 计算欧几里得距离
                    distance = math.sqrt(
                        (front_location.x - walker_location.x) ** 2 +
                        (front_location.y - walker_location.y) ** 2 +
                        (front_location.z - walker_location.z) ** 2
                    ) - front_vehicle_bounding_box.extent.x  # 计算的是中心距离，因此减去车的车头长度是合理的

                    # print(f"Distance is : {distance:.2f} 米")
                    # dis_x = abs(front_location.x - walker_location.x)
                    # print(f"Y distance is : {dis_x:.2f}m")

                    if 24< distance <= 31 :
                        if if_start == False:
                            event_start_frame = frame_total
                            print("\nevent_start_frame:",event_start_frame,"\n")
                            if_start = True
                        front_control.brake = 0.9
                        front_vehicle.apply_control(front_control)

                    if 16<distance <=24:
                        if if_start_cutin == False:
                            event_start_cutin = frame_total
                            print("\nevent_start_cut_in_frame:", event_start_cutin, "\n")
                            if_start_cutin = True

                        front_control.throttle = 0.5
                        front_control.steer = -0.2
                        front_vehicle.apply_control(front_control)
                    if 6< distance <=16:
                        front_control.steer = 0.2
                        front_control.throttle = 0.1
                        front_vehicle.apply_control(front_control)



                    if distance <= 2 or distance_ego <=1:
                        print('finish simulation!')
                        # resized_top_view_image = resize_image(top_view_image, 800, 600)
                        # cv2.imwrite(file_path+'raw_image/'+str(txt_number)+'.jpg', resized_top_view_image)
                        break



                if top_view_image is not None:
                    # 显示顶置俯视图像
                    resized_top_view_image = resize_image(top_view_image, 800, 600)
                    cv2.imshow('Top View Camera', resized_top_view_image)
                    cv2.putText(resized_top_view_image, 'frame: ' + str(frame_total), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # # 根据索引判断类别
                    # if predicted_index == 0:
                    #     cv2.putText(resized_top_view_image, 'Predict result: Safe --> ' + str(probabilities),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                    #     print("The prediction is: Safe")
                    # elif predicted_index == 1:
                    #     cv2.putText(resized_top_view_image, 'Predict result: Dangerous --> ' + str(probabilities),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                    #     print("The prediction is: Dangerous")
                    # else:
                    #     print("Unexpected index:", predicted_index)
                    #     cv2.putText(resized_top_view_image, 'Predict result: Dangerous --> ' + str(predicted_index),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                    final_img = resized_top_view_image
                    out.write(resized_top_view_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if keyboard.is_pressed('esc'):
                    print("检测到 'Esc' 键，退出程序。")
                    break

                # 计算时间差并更新行人的位置
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time<9:
                    elapsed_time=0
                else:
                    elapsed_time = elapsed_time-9
                # print('Elapsed_time: ',elapsed_time)
                new_y = walker_spawn_point.location.y + walker_speed * elapsed_time
                walker.set_location(carla.Location(
                    x=walker_spawn_point.location.x,
                    y=new_y,
                    z=walker_spawn_point.location.z
                ))
                # time.sleep(0.1)  # 控制更新频率

                try:
                    ############ >>>>>>>>>>>> TKG <<<<<<<<<<< ###########
                    #### 获取周围车辆信息
                    frame_participants_list=()   # 空元组
                    for vehicle in vehicles:
                        if vehicle.id == ego_vehicle.id:   # 排除本车
                            continue
                        vehicle_transform = vehicle.get_transform()  # Get the transform (position and orientation)
                        vehicle_location = vehicle_transform.location  # Extract the location
                        vehicle_velocity = vehicle.get_velocity()  # Get the velocity vector
                        velocity_other = 3.6 * math.sqrt(vehicle_velocity.x ** 2 + vehicle_velocity.y ** 2 + vehicle_velocity.z ** 2)

                        # acceleration_other = ego_vehicle.get_acceleration()  # 获取加速度
                        # print(f"加速度 (x, y, z): ({acceleration_other.x:.2f}, {acceleration_other.y:.2f}, {acceleration_other.z:.2f})")

                        distance = ego_location.distance(vehicle_location)     # 减去车道宽度
                        direction_vector = carla.Vector3D(
                            x=vehicle_location.x - ego_location.x,
                            y=vehicle_location.y - ego_location.y,
                            z=0  # 仅考虑平面上的夹角
                        )
                        angle = calculate_angle_between_vectors(ego_forward_vector, direction_vector)
                        # print(f"自车与前方车辆的夹角: {angle:.2f}°（以自车正前方为0°）")
                        # 将角度转换为弧度
                        angle_radians = math.radians(angle)
                        # 使用 sin(angle) * distance 计算水平方向的距离
                        horizontal_distance = math.sin(angle_radians) * distance
                        # print(f"基于角度和距离计算的水平距离: {horizontal_distance:.2f} 米")

                        # 根据水平距离判别出所在的车道信息,  阈值设定的原因来自Carla自身的车道宽度
                        if horizontal_distance >=3:
                            lane = "lane_4"
                        else:
                            if horizontal_distance <= -6.5:
                                lane = "lane_1"
                            elif  -6.5< horizontal_distance <= -3:
                                lane = "lane_2"
                            else:
                                lane = "lane_3"

                        # Print vehicle information
                        # print(f"Vehicle ID: {vehicle.id},{vehicle.type_id},{lane}")
                        # print(f"Location (x, y, z, distence): ({vehicle_location.x:.2f}, {vehicle_location.y:.2f}, {vehicle_location.z:.2f},{distance:.2f})")
                        # print(f"Velocity : ({velocity_other:.2f})")
                        # print("-" * 30)

                        ## 判别速度类别
                        velocity_other = round(velocity_other,1)
                        if velocity_other <= 1 :
                            speed_ = "static"
                        elif 1< velocity_other <= 10:
                            speed_ = "extremely-slow"
                        elif 10< velocity_other <= 20:
                            speed_ = "very-slow"
                        elif 20< velocity_other <= 30:
                            speed_ = "slow"
                        elif 30 < velocity_other <= 40:
                            speed_ = "low-normal"
                        elif 40< velocity_other <= 50:
                            speed_ = "normal"
                        elif 50< velocity_other <= 60:
                            speed_ = "high-normal"
                        elif 60 < velocity_other <= 80:
                            speed_ = "fast"
                        elif 80< velocity_other <= 100:
                            speed_ = "very-fast"
                        elif 100< velocity_other <= 120:
                            speed_ = "extremely-fast"
                        else:
                            speed_ = "dangerous-fast"

                        ## 判别方位
                        if 0<=angle <20:
                            ego_car_ = "front"
                        elif 20<=angle <60:
                            ego_car_ = "right_front"
                        elif 60<= angle:
                            ego_car_ = "right_side"
                        elif -20<=angle<0:
                            ego_car_ = "front"
                        elif -60<=angle<-20:
                            ego_car_ = "left_front"
                        else:
                            ego_car_ = "left_side"

                        # 取绝对值的角度
                        angle = abs(angle)
                        if 0<= angle <15:
                            angle_ = "angle-0-15"
                        elif 15<= angle <25:
                            angle_ = "angle-15-25"
                        elif 25<= angle <35:
                            angle_ = "angle-25-35"
                        elif 35<= angle <45:
                            angle_ = "angle-35-45"
                        elif 45<= angle <55:
                            angle_ = "angle-45-55"
                        elif 55<= angle <65:
                            angle_ = "angle-55-65"
                        elif 65<= angle <75:
                            angle_ = "angle-65-75"
                        elif 75<= angle <80:
                            angle_ = "angle-75-80"
                        elif 80<= angle <90:
                            angle_ = "angle-80-90"
                        else:
                            angle_ = "angle-90-180"

                        ## 判别距离
                        distance = abs(distance)
                        if 0<= distance <4:
                            distance_ = "dis-1"
                        elif 4<= distance <6:
                            distance_ = "dis-2"
                        elif 6<= distance <8:
                            distance_ = "dis-3"
                        elif 8<= distance <10:
                            distance_ = "dis-4"
                        elif 10<= distance <14:
                            distance_ = "dis-5"
                        elif 14<= distance <20:
                            distance_ = "dis-6"
                        elif 20<= distance <30:
                            distance_ = "dis-7"
                        elif 30<= distance <40:
                            distance_ = "dis-8"
                        else:
                            distance_ = "dis-9"

                        #########  行为判别
                        if frame_total % space_frame == 1:
                            if vehicle.id in behavior_state_name:
                                print(vehicle.id, 'is in behavior_state_name. ',
                                      behavior_state_name.index(vehicle.id))

                                if len(behavior_state_name_speed[behavior_state_name.index(vehicle.id)]) < 15:
                                    behavior_state_name_speed[behavior_state_name.index(vehicle.id)].append(
                                        velocity_other)
                                else:
                                    behavior_state_name_speed[behavior_state_name.index(vehicle.id)].pop(0)
                                    behavior_state_name_speed[behavior_state_name.index(vehicle.id)].append(
                                        velocity_other)
                            else:
                                behavior_state_name.append(vehicle.id)
                                behavior_state_name_speed.append([velocity_other, ])
                                print('behavior_state_name adding:', vehicle.id)

                        speed_dis = round(behavior_state_name_speed[behavior_state_name.index(vehicle.id)][-1] - behavior_state_name_speed[behavior_state_name.index(vehicle.id)][0],0)
                        # print(speed_dis)
                        if speed_dis<= -8:
                            behavior_ = 'hash_slow'
                        elif -8<speed_dis<=-2:
                            behavior_ = 'speed_down'
                        elif -2< speed_dis<=2:
                            behavior_ = 'uniform'
                        elif 2<speed_dis<8:
                            behavior_ = 'speed_up'
                        else:
                            behavior_ = 'hash_up'

                        frame_participants_list = frame_participants_list +(("v"+str(vehicle.id),"vehicle", "car",lane,"speed","position","angle",(speed_,angle_,distance_,ego_car_), behavior_),)
                    # print(behavior_state_name)
                    # print(behavior_state_name_speed)

                    #### 获取周围行人的信息
                    for pedestrian in pedestrians:
                        pedestrian_transform = pedestrian.get_transform()  # Get the transform (position and orientation)
                        pedestrian_location = pedestrian_transform.location  # Extract the location

                        distance_p = ego_location.distance(pedestrian_location)
                        direction_vector = carla.Vector3D(
                            x=pedestrian_location.x - ego_location.x,
                            y=pedestrian_location.y - ego_location.y,
                            z=0  # 仅考虑平面上的夹角
                        )
                        angle = calculate_angle_between_vectors(ego_forward_vector, direction_vector)
                        # print(f"自车与前方行人的夹角: {angle:.2f}°（以自车正前方为0°）")
                        # 将角度转换为弧度
                        angle_radians = math.radians(angle)
                        # 使用 sin(angle) * distance 计算水平方向的距离
                        horizontal_distance = math.sin(angle_radians) * distance_p
                        # print(f"基于角度和距离计算的水平距离: {horizontal_distance:.2f} 米")
                        # Print pedestrian information
                        # print(f"Pedestrian ID: {pedestrian.id}")
                        # print(
                        #     f"Location (x, y, z): ({pedestrian_location.x:.2f}, {pedestrian_location.y:.2f}, {pedestrian_location.z:.2f},{distance_p:.2f})")
                        # print("-" * 30)

                        ## 判别方位
                        if 0<=angle <20:
                            ego_p_ = "front"
                        elif 20<=angle <60:
                            ego_p_ = "right_front"
                        elif 60<= angle:
                            ego_p_ = "right_side"
                        elif -20<=angle<0:
                            ego_p_ = "front"
                        elif -60<=angle<-20:
                            ego_p_ = "left_front"
                        else:
                            ego_p_ = "left_side"
                        ## 判别角度
                        # 取绝对值的角度
                        angle = abs(angle)
                        if 0 <= angle < 15:
                            angle_ = "angle-0-15"
                        elif 15 <= angle < 25:
                            angle_ = "angle-15-25"
                        elif 25 <= angle < 35:
                            angle_ = "angle-25-35"
                        elif 35 <= angle < 45:
                            angle_ = "angle-35-45"
                        elif 45 <= angle < 55:
                            angle_ = "angle-45-55"
                        elif 55 <= angle < 65:
                            angle_ = "angle-55-65"
                        elif 65 <= angle < 75:
                            angle_ = "angle-65-75"
                        elif 75 <= angle < 80:
                            angle_ = "angle-75-80"
                        elif 80 <= angle < 90:
                            angle_ = "angle-80-90"
                        else:
                            angle_ = "angle-90-180"

                        ## 判别距离
                        distance = abs(distance_p)
                        if 0 <= distance < 4:
                            distance_ = "dis-1"
                        elif 4 <= distance < 6:
                            distance_ = "dis-2"
                        elif 6 <= distance < 8:
                            distance_ = "dis-3"
                        elif 8 <= distance < 10:
                            distance_ = "dis-4"
                        elif 10 <= distance < 14:
                            distance_ = "dis-5"
                        elif 14 <= distance < 20:
                            distance_ = "dis-6"
                        elif 20 <= distance < 30:
                            distance_ = "dis-7"
                        elif 30 <= distance < 40:
                            distance_ = "dis-8"
                        else:
                            distance_ = "dis-9"

                        if horizontal_distance <= 2:
                            lane_p = "lane_3"
                        else:
                            lane_p = "lane_4"

                        if distance <2:
                            print('*** Adding person to TKG! ***')
                            frame_participants_list = frame_participants_list +(("p"+str(pedestrian.id), "person", "man", lane_p, "speed","position","angle",( "fast",angle_,distance_,ego_p_), "uniform"),)
                    # print('current list: ',frame_participants_list)

                    ego_vehicle_velocity = ego_vehicle.get_velocity()  # Get the velocity vector
                    ego_velocity = 3.6 * math.sqrt(ego_vehicle_velocity.x ** 2 + ego_vehicle_velocity.y ** 2 + ego_vehicle_velocity.z ** 2)
                    ## 判别速度类别
                    if ego_velocity <= 1:
                        speed_ego = "static"
                    elif 1 < ego_velocity <= 10:
                        speed_ego = "extremely-slow"
                    elif 10 < ego_velocity <= 20:
                        speed_ego = "very-slow"
                    elif 20 < ego_velocity <= 30:
                        speed_ego = "slow"
                    elif 30 < ego_velocity <= 40:
                        speed_ego = "low-normal"
                    elif 40 < ego_velocity <= 50:
                        speed_ego = "normal"
                    elif 50 < ego_velocity <= 60:
                        speed_ego = "high-normal"
                    elif 60 < ego_velocity <= 80:
                        speed_ego = "fast"
                    elif 80 < ego_velocity <= 100:
                        speed_ego = "very-fast"
                    elif 100 < ego_velocity <= 120:
                        speed_ego = "extremely-fast"
                    else:
                        speed_ego = "dangerous-fast"


                    if frame_total % space_frame ==1:    ################ ############  每10帧增加一个关键帧
                        if len(traffic_participants_list)<10:
                            print('traffic_participants_list & ego_information_list is less than 15.')
                            traffic_participants_list = traffic_participants_list + (frame_participants_list,)

                            ego_information_list = ego_information_list + ( (("ego","vehicle","lane_3","speed",(speed_ego,"height","width")),), )

                        else:
                            print('traffic_participants_list length is 15.')
                            traffic_participants_list = traffic_participants_list[1:]
                            traffic_participants_list = traffic_participants_list + (frame_participants_list,)

                            ego_information_list = ego_information_list[1:]
                            ego_information_list = ego_information_list + ( (("ego", "vehicle", "lane_3", "speed", (speed_ego, "height", "width")),),)

                    scene = (road_structure_list,
                             road_facility_list,
                             road_change_list,
                             traffic_participants_list,
                             weather_list,
                             digital_information_list,
                             ego_information_list
                             )

                    # scene_length = scene_length_final   # 5个关键帧
                    #####  调用TKG生成GCN训练数据
                    if len(traffic_participants_list) >= 10 and frame_total % space_frame ==1 and frame_total>=500:
                        scene_final = scene
                        # pic_list.append(frame_total)
                        # pic_list.append(top_view_image)
                        scene_final_list.append(frame_total)
                        scene_final_list.append(scene_final)
                    #
                    #     TKG, tkg_node_feature_matrix, tkg_edge_index_list, tkg_edge_type_list, img_tkg = AVKG_3D_simple(
                    #         scene_final, scene_length_final, 4800, 4800, 4800,
                    #         is_visualization=False, is_export_triple=False, is_export_gnn=True
                    #     )
                    #     new_node_feature = torch.tensor(tkg_node_feature_matrix, dtype=torch.float).to(device)
                    #     new_edge_index = torch.tensor(tkg_edge_index_list, dtype=torch.long).t().to(
                    #         device)  # 转置，变为[2,n]
                    #     new_edge_attr = torch.tensor(tkg_edge_type_list, dtype=torch.long).to(device)
                    #     p = tkgcn(new_node_feature, new_edge_index, new_edge_attr)  # 图卷积内容
                    #     # 计算概率
                    #     probabilities = torch.softmax(p, dim=-1)
                    #     print('Result of risk prediction: ', probabilities)
                    #
                    #     # 获取概率最大的索引
                    #     predicted_index = torch.argmax(probabilities, dim=-1).item()
                    #     # 根据索引判断类别
                    #     if predicted_index == 0:
                    #         print("The prediction is: Safe")
                    #     elif predicted_index == 1:
                    #         print("The prediction is: Dangerous")
                    #     else:
                    #         print("Unexpected index:", predicted_index)



                except Exception as e:
                    print(f"Error in AVKG_3D_simple function: {e}")

            cv2.destroyAllWindows()

            out.release()# 释放视频写入对象
            print("Video saved successfully!")
            print('finish and we will visualize final graph!')

            probabilities_list_line = []
            time_taken_milliseconds_line = []

            dangerous_first_frame = 0
            if_dangerous_first_frame = False

            for i,item in enumerate(scene_final_list):
                if i % 2 == 1:
                    print("\nframe:",scene_final_list[i-1])
                    TKG, tkg_node_feature_matrix, tkg_edge_index_list, tkg_edge_type_list, img_tkg = AVKG_3D_simple(
                        item, scene_length_final, 4800, 4800, 4800,
                        is_visualization=False, is_export_triple=False, is_export_gnn=True
                    )
                    print('Current TRGCN data length is :', '\t')
                    print(len(tkg_node_feature_matrix), len(tkg_edge_index_list), len(tkg_edge_type_list))
                    ############ >>>>>>>>>>>> TRGCN <<<<<<<<<<< ###########
                    # 将 numpy.ndarray 转换为 torch.tensor
                    new_node_feature = torch.tensor(tkg_node_feature_matrix, dtype=torch.float).to(device)
                    new_edge_index = torch.tensor(tkg_edge_index_list, dtype=torch.long).t().to(device)  # 转置，变为[2,n]
                    new_edge_attr = torch.tensor(tkg_edge_type_list, dtype=torch.long).to(device)
                    # print(new_edge_index)
                    t_predict_front = time.time()
                    p = tkgcn(new_node_feature, new_edge_index, new_edge_attr)  # 图卷积内容
                    t_predict_after = time.time()
                    time_taken_milliseconds = (t_predict_after - t_predict_front) * 1000
                    time_taken_milliseconds_line.append(time_taken_milliseconds)
                    print("This frame risk predict time is:", time_taken_milliseconds, "milliseconds")
                    # 计算概率
                    probabilities = torch.softmax(p, dim=-1)
                    print('Result of risk prediction: ', probabilities)

                    # 获取概率列表（去除计算图并将其转移到 CPU）
                    probabilities_list = probabilities.cpu().detach().numpy().tolist()

                    # final_img_ = np.ones((600, 1000, 3), dtype=np.uint8) * 255

                    # 获取概率最大的索引
                    predicted_index = torch.argmax(probabilities, dim=-1).item()
                    probabilities_list_line.append((scene_final_list[i-1],probabilities_list[0]))
                    # 根据索引判断类别
                    if predicted_index == 0:
                        print("The prediction is: Safe",probabilities_list)
                        # cv2.putText(final_img_, 'Predict result: Safe' + str(probabilities_list), (10, 640),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    elif predicted_index == 1:
                        print("The prediction is: Dangerous",probabilities_list)
                        if if_dangerous_first_frame == False:
                            dangerous_first_frame = scene_final_list[i-1]
                            if_dangerous_first_frame = True
                        # cv2.putText(final_img_, 'Predict result: Dangerous' + str(probabilities_list), (10, 640),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    else:
                        print("Unexpected index:", predicted_index)

                    if predicted_index == 1:
                        print('Start EKG inference: ')
                        risk = AVKG_3D_simple_EKG(TKG, is_draw_3d=False)
                        print("Risk source is :", risk)

                    # if predicted_index == 1:
                    #     print('Start EKG inference: ')
                    #     risk, img_EKG = AVKG_3D_simple_EKG(TKG)
                    #     print("Risk source is :", risk)
                    #     cv2.putText(final_img_, "Risk source is :" + risk, (10, 700),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    # final_img_[0:600, 0:800] = final_img
                    #
                    #
                    # cv2.imshow('final', pic_list[i])
                    # # 等待键盘事件，按任意键关闭窗口
                    # cv2.imwrite('result/'+str(i)+'-result.jpg', pic_list[i])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()



            TKG, tkg_node_feature_matrix, tkg_edge_index_list, tkg_edge_type_list, img_tkg = AVKG_3D_simple(
                scene_final, scene_length_final, 4800, 4800, 4800,
                is_visualization=True, is_export_triple=False, is_export_gnn=True
            )
            print('Current TRGCN data length is :', '\t')
            print(len(tkg_node_feature_matrix), len(tkg_edge_index_list), len(tkg_edge_type_list))
            ############ >>>>>>>>>>>> TRGCN <<<<<<<<<<< ###########
            # 将 numpy.ndarray 转换为 torch.tensor
            new_node_feature = torch.tensor(tkg_node_feature_matrix, dtype=torch.float).to(device)
            new_edge_index = torch.tensor(tkg_edge_index_list, dtype=torch.long).t().to(device)  # 转置，变为[2,n]
            new_edge_attr = torch.tensor(tkg_edge_type_list, dtype=torch.long).to(device)
            # print(new_edge_index)
            t_predict_front = time.time()
            p = tkgcn(new_node_feature, new_edge_index, new_edge_attr)  # 图卷积内容
            t_predict_after = time.time()
            time_taken_milliseconds = (t_predict_after - t_predict_front) * 1000
            print("This frame risk predict time is:", time_taken_milliseconds, "milliseconds")
            # 计算概率
            probabilities = torch.softmax(p, dim=-1)
            print('Result of risk prediction: ', probabilities)

            # 获取概率列表（去除计算图并将其转移到 CPU）
            probabilities_list = probabilities.cpu().detach().numpy().tolist()

            # 获取概率最大的索引
            predicted_index = torch.argmax(probabilities, dim=-1).item()
            # 根据索引判断类别
            if predicted_index == 0:
                print("The prediction is: Safe")
                cv2.putText(final_img_all, 'Predict result: Safe' + str(probabilities_list), (10, 640),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            elif predicted_index == 1:
                print("The prediction is: Dangerous")
                cv2.putText(final_img_all, 'Predict result: Dangerous' + str(probabilities_list), (10, 640),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                print("Unexpected index:", predicted_index)

            if predicted_index == 1:
                print('Start EKG inference: ')
                risk, img_EKG = AVKG_3D_simple_EKG(TKG,is_draw_3d=True)
                print("Risk source is :", risk)
                cv2.putText(final_img_all, "Risk source is :" + risk, (10, 700),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                final_img_all[0:600, 0:800] = final_img
                final_img_all[:600, 800:] = img_tkg
                final_img_all[600:, 800:] = img_EKG

                cv2.imshow('final', final_img_all)
                # 等待键盘事件，按任意键关闭窗口
                cv2.imwrite('result/cutin_result-1-19.jpg', final_img_all)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            ########## 绘制图像 ###########
            print(probabilities_list_line)
            print(time_taken_milliseconds_line)
            x = [item[0] for item in probabilities_list_line]  # 横坐标：每个元组的第一个元素
            y = [item[1][1] for item in probabilities_list_line]  # 纵坐标：每个元组第二个元素的索引1的值

            # 绘制折线图
            plt.figure(figsize=(18, 6))
            plt.plot(x, y, marker='', color='r', linestyle='-', label='Probability of potential risk')
            # 填充折线下方区域为浅红色
            plt.fill_between(x, y, color='red', alpha=0.2)
            # 设置图形标题和标签
            plt.title("Probability of potential risk throughout the event")
            plt.xlabel("Frame i")
            plt.ylabel("Probability of potential risk")

            x_position = event_start_frame

            # 绘制蓝色虚线
            plt.axvline(x=x_position, color='black', linestyle='-', label=f"Event start at frame x={x_position}")

            cutin_position = event_start_cutin

            # 绘制蓝色虚线
            plt.axvline(x=cutin_position, color='blue', linestyle='--', label=f"Start cut in at frame x={cutin_position}")

            # 绘制蓝色虚线
            plt.axvline(x=dangerous_first_frame, color='red', linestyle='--',
                        label=f"Detect dangerous at frame x={dangerous_first_frame}")

            # 显示网格
            plt.grid(False)

            # 显示图例
            plt.legend()

            # 保存图像到本地
            plt.savefig('result/line_plot_cutin-1-19.png')

            # 展示图像
            plt.show()







        except KeyboardInterrupt:
            print("\n检测到退出信号，正在销毁车辆...")
        finally:
            # 清理资源
            top_view_camera.stop()
            top_view_camera.destroy()
            walker.destroy()
            ego_vehicle.destroy()
            front_vehicle.destroy()
            # other_vehicle.destroy()
            # other_vehicle_2.destroy()
            # other_vehicle_3.destroy()
            # other_vehicle_4.destroy()
            cv2.destroyAllWindows()
            print("车辆已销毁。程序结束。")


    except Exception as e:
        print(f"发生错误: {e}")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
