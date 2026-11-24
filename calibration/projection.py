"""Project world-space calibration faces using the actual recorded client camera."""
import numpy as np

def camera_project(xyz,pose):
    yaw,pitch=np.radians(pose['yaw_pitch_deg'])
    forward=np.array([-np.sin(yaw)*np.cos(pitch),-np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    right=np.array([-np.cos(yaw),0,-np.sin(yaw)])
    up=np.cross(right,forward)
    rel=np.asarray(xyz)-pose['eye_world']
    w,h=pose['image_size']
    focal=h/(2*np.tan(np.radians(pose['vertical_fov_deg'])/2))
    return np.stack([w/2+focal*(rel@right)/(rel@forward),h/2-focal*(rel@up)/(rel@forward)],axis=-1)


def south_quad(x,y,z):
    return np.array([[x,y+1,z+1],[x+1,y+1,z+1],[x+1,y,z+1],[x,y,z+1]],dtype=float)
