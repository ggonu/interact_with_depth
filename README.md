# interact_with_depth
My playground play with RealSense!!

Update soon.

# Dependencies

# Demo
```(bash)
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30 pointcloud.enable:=true align_depth:=true
```

```(bash)
# In the new terminal
ros2 run interact_with_depth depthProcess
```