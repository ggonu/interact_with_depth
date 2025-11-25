/**
 * @author Kho Geonwoo
 */

#pragma once

#include <random>
#include <vector>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>


using PointT = pcl::PointXYZ;

class depthTrackingProcess : public rclcpp::Node {
private:
  struct Object {
    int id;
    float cx, cy, cz;
    uint8_t r, g, b;
    rclcpp::Time last_seen;
  };

  struct Cluster {
    std::vector<int> indices;
    float cx, cy, cz;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    int track_id;
    uint8_t r, g, b;
  };

  void pcCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg
  );

  void filterCloud(
    const pcl::PointCloud<PointT>::Ptr &cloud_in,
    pcl::PointCloud<PointT>::Ptr &cloud_out
  );

  void euClustering(
    const pcl::PointCloud<PointT>::Ptr &cloud,
    std::vector<pcl::PointIndices> &cluster_indices
  );

  void computeClusters(
    const pcl::PointCloud<PointT>::Ptr &cloud,
    const std::vector<pcl::PointIndices> &cluster_indices,
    std::vector<Cluster> &clusters
  );

  void updateTracks(
    std::vector<Cluster> &clusters,
    const rclcpp::Time &stamp
  );

  void buildOutput(
    const pcl::PointCloud<PointT>::Ptr &cloud,
    const std::vector<Cluster> &clusters,
    const std_msgs::msg::Header &header
  );

  // === Params
  std::string input_topic_;
  std::string cloud_pub_topic_;
  std::string marker_pub_topic_;

  double max_range_;
  double cluster_tolerance_;
  int    min_cluster_size_;
  int    max_cluster_size_;
  double association_distance_;
  double track_timeout_sec_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::vector<Object> objects_;
  int next_track_id_;
  std::mt19937 rng_;

public:
  depthTrackingProcess();

};