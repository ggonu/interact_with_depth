/**
 * @author Kho Geonwoo
 */

#pragma once

#include <string>

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

using PointT = pcl::PointXYZ;

class depthProcess : public rclcpp::Node {
private:
  std::string input_topic_;
  std::string cloud_pub_topic_;
  std::string marker_pub_topic_;

  // Segmentation parameters
  double sac_distance_threshold_;
  int sac_max_iter_;

  // Clustering parameters
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  void pcCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg
  );

  void filterMaxRange(
    const pcl::PointCloud<PointT>::Ptr &cloud_in,
    pcl::PointCloud<PointT>::Ptr &cloud_out
  );

  void rmGroundPlane(
    const pcl::PointCloud<PointT>::Ptr &cloud_in,
    pcl::PointCloud<PointT>::Ptr &cloud_out
  );

  void euClustering(
    const pcl::PointCloud<PointT>::Ptr &cloud,
    std::vector<pcl::PointIndices> &cluster_indices
  );

  /**
   * @brief Build egmented point cloud and markers(visualization_msgs::msg::MarkerArray)
   */
  void buildSegments(
    const pcl::PointCloud<PointT>::Ptr &cloud,
    const std::vector<pcl::PointIndices> &cluster_indices,
    const std_msgs::msg::Header &header,
    sensor_msgs::msg::PointCloud2 &segmented_msg,
    visualization_msgs::msg::MarkerArray &marker_array
  );


public:
  depthProcess();

};

// depthProcess