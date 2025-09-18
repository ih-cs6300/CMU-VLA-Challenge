import time
import rospy
import os
import numpy as np

from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker

def publish_waypoints(self, path_ordered, publisher, threshold=0.6):
    for wp_num, waypoint in enumerate(path_ordered):
        print(f"  wp_num: {wp_num}/{len(path_ordered)-1}")
        #rospy.loginfo(f"waypoint: ({waypoint[0]}, {waypoint[1]})")
        t1 = time.time()
        pose_msg = Pose2D(x=waypoint[0], y=waypoint[1], theta=0.)
        publisher.publish(pose_msg)
        #os.system(f"rostopic pub -1 /way_point_with_heading geometry_msgs/Pose2D '{{x: {waypoint[0]}, y: {waypoint[1]}, theta: 0.0}}'") 
        diff = np.array([self.position_x, self.position_y]) - np.array([waypoint[0], waypoint[1]])
        dist_wp = np.linalg.norm(diff)  # L2 distance
        delta_t = time.time() - t1
        while(dist_wp > threshold) and (delta_t < 12):
            time.sleep(2)  # sleep 1 second
            delta_t = time.time() - t1
            diff = np.array([self.position_x, self.position_y]) - np.array([waypoint[0], waypoint[1]])
            dist_wp = np.linalg.norm(diff)
            #print(f"delta_t: {delta_t}")
            #print(f"dist_wp: {dist_wp}")


def publish_numerical(self, num_answer, publisher):
    int_msg = Int32()
    int_msg.data = num_answer
    publisher.publish(int_msg)


def publish_marker(self, marker_answer, publisher):
    marker_answer = self.gdf.loc[self.gdf.geometry == marker_answer, :].iloc[0]
    # Create a Marker message
    marker = Marker()
    marker.header.frame_id = "map"  # Reference frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = marker_answer['object_name']
    marker.id = 0
    marker.type = 1 # Assuming a cube for bounding box
    marker.action = 0

    # Set the position of the marker
    marker.pose.position.x = marker_answer['geometry'].x
    marker.pose.position.y = marker_answer['geometry'].y
    marker.pose.position.z = marker_answer['geometry'].z
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Set the scale (bounding box size)
    marker.scale.x = marker_answer['scale'][0]
    marker.scale.y = marker_answer['scale'][1]
    marker.scale.z = marker_answer['scale'][2]

    # Set the color of the marker
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.4  # Alpha (transparency)

    publisher.publish(marker)
