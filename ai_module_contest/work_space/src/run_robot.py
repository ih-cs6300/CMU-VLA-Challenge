#!/usr/bin/env python3

import rospy
import numpy as np
import struct
import matplotlib.pyplot as plt
import tf.transformations # Used for converting quaternions to Euler angles
import math
import os
import time

from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField # Import PointField for constants
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2

from scipy import ndimage
from tsp_solver.greedy import solve_tsp
from scipy.spatial import KDTree

import torch

import geopandas as gpd
from shapely.geometry import Point

from path_utils import reduce_path, clean_path, smooth_path
from process_question import question_processor_driver, get_qtype_objects
from publish_answers import publish_waypoints, publish_numerical, publish_marker

class RobotProcessor:
    """
    robot object
    """
    def __init__(self, init_state=0):
        """
        Initializes the RobotProcessor node, setting up the subscriber.
        """
        rospy.init_node('robo_processor_node', anonymous=True)
        #rospy.loginfo("RobotProcessor node initialized.")

        # Subscribe to the 'traversable_area' topic
        self.trav_subscr = rospy.Subscriber(

           '/traversable_area',
           #'/my_semantic_scan',
            PointCloud2,
            self._pointcloud_callback
        )
        #rospy.loginfo("Subscribed to topic: /traversable_area of type sensor_msgs/PointCloud2")

        self.odom_subscr = rospy.Subscriber(
            '/state_estimation',
            Odometry, 
            self._odometry_callback
        )
        #rospy.loginfo("Subscribed to /state_estimation topic...")
        
        self.marker_subscr = rospy.Subscriber(
            '/object_markers',
            MarkerArray, 
            self._marker_callback
        )
        #rospy.loginfo("Subscribed to /object_markers topic...")
        #self.sem_img_subscr = rospy.Subscriber(
        #    '/camera/semantic_image',
        #    #'/camera/image', 
        #    Image, 
        #    self._image_callback
        #)
        #rospy.loginfo("Subscribed to /camera/image topic...")

        #self.reg_scan_subscr = rospy.Subscriber(
        #    '/registered_scan',
        #    PointCloud2,
        #    self._regscan_callback
        #)
        #rospy.loginfo("Subscribed to /registered_scan topic...")

        self.question_subscr = rospy.Subscriber(
            '/challenge_question',
            String,
            self._challenge_callback
        )

        self.waypoint_pub = rospy.Publisher('/way_point_with_heading', Pose2D, queue_size=1000)
        self.numerical_pub = rospy.Publisher('/numerical_response', Int32, queue_size=10)
        self.marker_pub = rospy.Publisher('/selected_object_marker', Marker, queue_size=10)
        

        self.model = None
        self.position_x = -99.
        self.position_y = -99.
        self.position_z = -99.
        self.linear_x = -99.
        self.linear_y = -99.
        self.linear_z = -99.

        self.point_matrix = None

        # these flags control state transition
        self.got_position = False
        self.got_traversable = False
        self.got_question = False
        self.done_exploring = False 
        self.got_answer = False

        # robot state
        self.state = init_state                                                                         
        self.state_names = ['initialize', 'explore', 'process_question', 'answer_question', 'done']
        self.old_state = -1

        # challenge question
        self.statement_type = ""
        self.challenge_question = ""
        self.answer = ""

        self.object_markers = {}

        self.bridge = CvBridge()

        spatial_data = {"id":[], "object_name":[], "geometry":[], "scale":[], "orientation":[], "text_embedding":[]}
        self.gdf = gpd.GeoDataFrame(spatial_data, geometry='geometry')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _challenge_callback(self, msg):
        #rospy.loginfo("String Processor Node: Received Question: '%s'", msg.data) 

        if (self.got_question == False) and (msg.data is not None):
            self.challenge_question = msg.data
            self.got_question = True
            get_qtype_objects(self, self.challenge_question)

    def _process_question(self):
        question_processor_driver(self)
        self.got_answer = True

    def _pointcloud_callback(self, msg):
        """
        Callback function for the /traversable_area topic.
        This function is called every time a new PointCloud2 message is received.

        Args:
            msg (sensor_msgs.msg.PointCloud2): The received PointCloud2 message.
        """
        #rospy.loginfo("Received PointCloud2 message.")

        # Convert the PointCloud2 message to a NumPy matrix
        point_matrix = self._convert_pointcloud2_to_numpy(msg)

        if point_matrix is not None:
            #rospy.loginfo(f"Successfully converted PointCloud2 to NumPy Matrix. Shape: {point_matrix.shape}")
            # You can now process 'point_matrix' further, e.g.,
            # - Extract X, Y, Z coordinates: point_matrix[:, 0], point_matrix[:, 1], point_matrix[:, 2]
            # - Perform filtering, clustering, etc.
            
            # For demonstration, print the first 5 points if available
            if point_matrix.shape[0] > 0:
                #rospy.loginfo("\nFirst 5 points (or fewer if less than 5):")
                #rospy.loginfo(point_matrix[:min(5, point_matrix.shape[0])])
                np.save('/home/ubuntu/CMU-VLA-Challenge/occ_grid99.npy', point_matrix)

                # store point matrix
                self.point_matrix = point_matrix

                # set flag for state change
                if (self.got_traversable == False):
                    self.got_traversable = True

            else:
                rospy.loginfo("Matrix is empty (no points).")
        else:
            rospy.logwarn("Failed to convert PointCloud2 message to NumPy matrix.")

    def _convert_pointcloud2_to_numpy(self, pc2_msg):
        """
        Converts a sensor_msgs/PointCloud2 message into a NumPy matrix.

        This function parses the raw byte data based on the PointField definitions
        to reconstruct the points.

        Args:
            pc2_msg (sensor_msgs.msg.PointCloud2): The ROS PointCloud2 message.

        Returns:
            numpy.ndarray: A NumPy array where each row represents a point and
                           columns correspond to the fields defined in the message.
                           Returns None if conversion fails or data is malformed.
        """
        if not pc2_msg or not pc2_msg.data:
            rospy.logwarn("Empty or invalid PointCloud2 message data.")
            return None

        num_points = pc2_msg.width * pc2_msg.height
        if num_points == 0:
            #rospy.loginfo("PointCloud2 message contains no points (num_points = 0).")
            return np.empty((0, len(pc2_msg.fields)))

        # Determine byte order for struct unpacking
        # '<' for little-endian, '>' for big-endian
        endian_prefix = '>' if pc2_msg.is_bigendian else '<'

        # List to hold all points, which will be converted to a NumPy array
        all_points = []

        try:
            # Iterate over each point in the data blob
            for i in range(num_points):
                # Calculate the start and end byte index for the current point
                point_data_start = i * pc2_msg.point_step
                point_data_end = point_data_start + pc2_msg.point_step
                
                # Extract the byte slice for the current point
                current_point_bytes = pc2_msg.data[point_data_start:point_data_end]

                point_values = []
                # Iterate over each field within the current point
                for field in pc2_msg.fields:
                    # Determine the correct struct format character based on PointField datatype
                    fmt_char = ''
                    if field.datatype == PointField.INT8:    fmt_char = 'b'
                    elif field.datatype == PointField.UINT8:  fmt_char = 'B'
                    elif field.datatype == PointField.INT16:   fmt_char = 'h'
                    elif field.datatype == PointField.UINT16:  fmt_char = 'H'
                    elif field.datatype == PointField.INT32:   fmt_char = 'i'
                    elif field.datatype == PointField.UINT32:  fmt_char = 'I'
                    elif field.datatype == PointField.FLOAT32: fmt_char = 'f'
                    elif field.datatype == PointField.FLOAT64: fmt_char = 'd'
                    else:
                        rospy.logwarn(f"Unsupported datatype {field.datatype} for field {field.name}. Skipping this field.")
                        # Append NaN or a default value for unsupported types to maintain column count
                        point_values.extend([np.nan] * field.count) 
                        continue

                    # Unpack each element for the current field (field.count is usually 1)
                    for k in range(field.count):
                        # Calculate the offset within the current point's byte slice
                        field_start_offset = field.offset + k * struct.calcsize(fmt_char)
                        
                        # Unpack the value. struct.unpack_from handles the offset
                        value = struct.unpack_from(endian_prefix + fmt_char, current_point_bytes, field_start_offset)[0]
                        point_values.append(value)
                
                all_points.append(point_values)

            # Convert the list of lists into a NumPy array
            # Use float32 as a common and efficient type for point cloud coordinates
            return np.array(all_points, dtype=np.float32)

        except struct.error as e:
            rospy.logerr(f"Struct unpacking error during PointCloud2 conversion: {e}. Check message integrity or field definitions.")
            return None
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred during PointCloud2 conversion: {e}")
            return None

    def _odometry_callback(self, msg):
        """
        Callback function for the /state_estimation topic.
        Processes the incoming Odometry messages.
        """
        #rospy.loginfo("--- Received Odometry Message ---")
        # Extract position (x, y)
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z

        # Extract orientation (quaternion)
        orientation_q = msg.pose.pose.orientation
        # ROS quaternions are typically stored as (x, y, z, w)
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # The output is in radians
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        # Print the extracted pose information
        #rospy.loginfo("Position: X=%.2f m, Y=%.2f m", self.position_x, self.position_y)
        #rospy.loginfo("Orientation (Yaw): %.2f radians (%.2f degrees)", yaw, math.degrees(yaw))

        # You can also access twist (linear and angular velocities) if needed
        self.linear_x = msg.twist.twist.linear.x
        self.linear_y = msg.twist.twist.linear.y
        #rospy.loginfo("Linear Velocity: X=%.2f m/s", self.linear_x)
        #rospy.loginfo("Linear Velocity: Y=%.2f m/s", self.linear_y)

        # set flags for state change
        if (self.got_position == False):
            self.got_position = True

    def _marker_callback(self, msg):
        #rospy.loginfo("Received a MarkerArray message with %d markers.", len(msg.markers))

        for i, marker in enumerate(msg.markers):
            #rospy.loginfo("--- Marker %d ---", i + 1)
            #rospy.loginfo("  Header: stamp=%s, frame_id=%s", marker.header.stamp, marker.header.frame_id)
            #rospy.loginfo("  ID: %d", marker.id)
            #rospy.loginfo("  Type: %d", marker.type)
            #rospy.loginfo("  Action: %d", marker.action)
            #rospy.loginfo("  Pose: position=(%.2f, %.2f, %.2f), orientation=(%.2f, %.2f, %.2f, %.2f)",
            #              marker.pose.position.x, marker.pose.position.y, marker.pose.position.z,
            #              marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w)
            #rospy.loginfo("  Scale: (%.2f, %.2f, %.2f)", marker.scale.x, marker.scale.y, marker.scale.z)
            #rospy.loginfo("  Color: rgba=(%.2f, %.2f, %.2f, %.2f)", marker.color.r, marker.color.g, marker.color.b, marker.color.a)
            #rospy.loginfo("  Lifetime: %s seconds", marker.lifetime.secs)
            # You can add more detailed processing here based on your specific needs for each marker.

            # catalog markers by name and id
            marker_position = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            marker_scale = (marker.scale.x, marker.scale.y, marker.scale.z)
            marker_orientation = (marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w)
            if (self.gdf[(self.gdf.object_name == marker.ns) & (self.gdf.geometry == Point(marker_position))].empty):
                # add row to dataframe
                self.gdf.loc[len(self.gdf)] = [marker.id, marker.ns, Point(marker_position), marker_scale, marker_orientation, 999]
                print(f"    marker: {marker.ns}")


    def _image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (NumPy array)
            # 'bgr8' is often used for color images, 'mono8' for grayscale
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Now cv_image is a NumPy array. You can perform operations on it.
            # For example, print its shape and data type:
            #rospy.loginfo(f"Image converted to NumPy array. Shape: {cv_image.shape}, Data Type: {cv_image.dtype}")


        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        # Generate a unique filename using timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        #filename = os.path.join(self.output_directory, f"image_{timestamp}_{self.image_count:04d}.png")
        filename = "/home/ubuntu/CMU-VLA-Challenge/semantic_image1.png"

        try:
            # Save the OpenCV image as a PNG file
            cv2.imwrite(filename, cv_image)
            #rospy.loginfo(f"Saved image: {filename}")
        except Exception as e:
            rospy.logerr(f"Error saving image {filename}: {e}")


    def _regscan_callback(self, msg):
        point_matrix = self._convert_pointcloud2_to_numpy(msg)
        if point_matrix.shape[0] > 0:
            #rospy.loginfo("\nFirst 5 points (or fewer if less than 5):")
            #rospy.loginfo(point_matrix[:min(5, point_matrix.shape[0])])
            np.save('/home/ubuntu/CMU-VLA-Challenge/regscan_grid.npy', point_matrix)

    def save_gdf(self):
        self.gdf.to_pickle('gpandas_df.pkl')        

    def state_controller(self, event):
        # state controller 

        if (self.state == 0):
            # init
            if (self.state != self.old_state):
                print(f"\nstate: {self.state_names[self.state]}")
                self.old_state = self.state

            if (self.got_position == True) and (self.got_traversable == True) and (self.got_question == True) and (self.statement_type != ""): 
                self.state = 1
                print(f"\nstate: {self.state_names[self.state]}")
                if self.statement_type != 'instruction-following':
                    self._explore1()
                else:
                    self._explore1()
        elif (self.state == 1):
            # explore
            if (self.done_exploring == True) and (self.got_question == True): 
                self.state = 2
                print(f"\nstate: {self.state_names[self.state]}")
                self.unsubscribe_markers();
                self._process_question()
        elif (self.state == 2):
            # process question
            if (self.got_answer == True):
                self.state = 3
                print(f"\nstate: {self.state_names[self.state]}")
        elif (self.state == 3):
            # answer question
            if (self.statement_type == 'numerical'):
                print(f"answer: {self.answer}")
                self.state = 4
                print(f"\nstate: {self.state_names[self.state]}")
            elif (self.statement_type == 'object-reference'):
                print(f"answer: {self.answer}")
                self.state = 4
                print(f"\nstate: {self.state_names[self.state]}")
            elif (self.statement_type == 'instruction-following'):
                print(f"answer: {self.answer}")
                publish_waypoints(self, np.array(self.answer), self.waypoint_pub)
                self.state = 4
                print(f"\nstate: {self.state_names[self.state]}") 
            else:
                raise ExceptionType('bad statement type')
        elif (self.state == 4):
            # done
            pass

    def unsubscribe_markers(self):
        # unsubscribe to marker
        self.marker_subscr.unregister()

    def explore_traversable(self, xedges, yedges, counts, minx, maxx, miny, maxy, start_pt):
        # get center of each box
        cell_width = xedges[1] - xedges[0]
        cell_height = yedges[1] - yedges[0]

        mid_width = cell_width/2
        mid_height = cell_height/2

        xcenters = xedges + mid_width
        ycenters = yedges + mid_height

        # remove last edges
        xcenters = xcenters[:len(xcenters)-1]
        ycenters = ycenters[:len(ycenters)-1]

        Xcenters, Ycenters = np.meshgrid(xcenters, ycenters, indexing='ij')

        # remove cells that contain obstacle or are too close to an obstacle
        counts = np.where(counts > 10., 1., 0.)                                         # counts > 1. => traversable

        dist_transform = ndimage.distance_transform_edt(np.int64(np.bool_(counts)))   # foreground = 1
        dist_transform *= 0.25
        Xfiltered = Xcenters[np.where(dist_transform > 0.3)]
        Yfiltered = Ycenters[np.where(dist_transform > 0.3)]


        centers_matrix = np.concatenate([Xfiltered.flatten().reshape(-1, 1), Yfiltered.flatten().reshape(-1, 1)], axis=1)

        centers_matrix = centers_matrix[((minx < centers_matrix[:, 0]) & (centers_matrix[:, 0] < maxx)) & ((miny < centers_matrix[:, 1]) & ( centers_matrix[:, 1] < maxy))]

        # calculate distance between points
        z = np.array([[complex(*c) for c in centers_matrix]])
        distance_matrix = abs(z.T-z)


        # find point closest to start
        start_idx = np.argmin(np.linalg.norm(centers_matrix - np.array(start_pt), axis=1, ord=2))

        # solve for roundtrip path
        path = solve_tsp(distance_matrix, endpoints=(start_idx, start_idx))

        return path, centers_matrix


    def _explore1(self):
        #rospy.loginfo("----------goto_point------------")

        buffer = 0
        p1 = (self.position_x, self.position_y)
        point_matrix = self.point_matrix
        kdtree = KDTree(point_matrix[:, :2])
        MIN_X = np.min(point_matrix[:, 0]) + buffer
        MAX_X = np.max(point_matrix[:, 0]) - buffer
        MIN_Y = np.min(point_matrix[:, 1]) + buffer
        MAX_Y = np.max(point_matrix[:, 1]) - buffer 
        area_est = (MAX_X - MIN_X) * (MAX_Y - MIN_Y)

        counts, xedges, yedges = np.histogram2d(point_matrix[:, 0], point_matrix[:, 1], bins=( np.ceil((MAX_Y-MIN_Y)/0.25).astype(np.int64), np.ceil((MAX_X-MIN_X)/0.25).astype(np.int64) )) 
        # exploration using tsp
        exploration_path, c_matrix = self.explore_traversable(xedges, yedges, counts, MIN_X, MAX_X, MIN_Y, MAX_Y, p1)
        path_ordered = c_matrix[exploration_path].tolist()

        counts = np.where(counts > 10., 0., 1.)

        if (area_est >= 0) and (area_est <= 42):
            n_pts = 35
        elif (area_est > 42) and (area_est < 60):
            n_pts = 50
        else:
            n_pts = 150

        # process path
        new_path = clean_path(path_ordered, kdtree, point_matrix, xedges, yedges, counts)
        new_path = smooth_path(new_path, n_pts, s_var=4) # 25
        new_path = clean_path(new_path, kdtree, point_matrix, xedges, yedges, counts)
        path_ordered = np.array(reduce_path(new_path))

        publish_waypoints(self, path_ordered, self.waypoint_pub)
        self.done_exploring = True

    def _explore2(self):
        # exploration for instruction_following 
        # don't move since all movement is interpreted as part of answer
        time.sleep(20)
        pos = np.array([self.position_x, self.position_y])
        point_matrix = self.point_matrix
        kdtree = KDTree(point_matrix[:, :2])
        MIN_X = np.min(point_matrix[:, 0])
        MAX_X = np.max(point_matrix[:, 0])
        MIN_Y = np.min(point_matrix[:, 1])
        MAX_Y = np.max(point_matrix[:, 1])

        counts, xedges, yedges = np.histogram2d(point_matrix[:, 0], point_matrix[:, 1], bins=( np.ceil((MAX_Y-MIN_Y)/0.25).astype(np.int64), np.ceil((MAX_X-MIN_X)/0.25).astype(np.int64) ))
        counts = np.where(counts > 10., 0., 1.)

        # go in square
        p1 = pos + np.array([1.2, 0.])
        p2 = p1 + np.array([-1.2, 0.])

        dist1, idx1 = kdtree.query(p1)
        dist2, idx2 = kdtree.query(p2)

        p1 = point_matrix[idx1, :2]
        p2 = point_matrix[idx2, :2]

        path_ordered = np.array([p1, p2])
        print(path_ordered)
        publish_waypoints(self, path_ordered, self.waypoint_pub, threshold=0.1)

        self.done_exploring = False


def create_shutdown_hook(process_obj):
    rospy.loginfo("Node is shutting down. Performing cleanup...")
    def shutdown_hook_with_data():
        print("saving files...")
        process_obj.save_gdf()
    return shutdown_hook_with_data

def main():
    """
    Main function to create and run the PointCloudProcessor node.
    """
    try:
        processor = RobotProcessor(init_state=0)
        rospy.on_shutdown(create_shutdown_hook(processor))
        rospy.Timer(rospy.Duration(1.0), processor.state_controller)  # every second
        rospy.spin() # Keep the node running and processing callbacks

    except rospy.ROSInterruptException:
        rospy.loginfo("RobotProcessor node shut down.")


if __name__ == '__main__':
    main()

