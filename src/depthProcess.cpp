/**
 * @author Kho Geonwoo
 */

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include "interact_with_depth/depthProcess.hpp"

void depthProcess::pcCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg
) {
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
  pcl::fromROSMsg(*msg, *cloud);

  RCLCPP_INFO(
    this->get_logger(),
    "Received PointCloud2: width=%u height=%u size=%zu",
    msg->width, msg->height, cloud->points.size()
  );

  if (cloud->empty()) {
    RCLCPP_INFO(this->get_logger(), "Input PointCloud msg is EMPTY.");
    return;
  }

  // // Ground plane removal
  // pcl::PointCloud<PointT>::Ptr cloud_roi(new pcl::PointCloud<PointT>());
  // rmGroundPlane(cloud, cloud_roi);

  pcl::PointCloud<PointT>::Ptr cloud_roi(new pcl::PointCloud<PointT>());
  filterMaxRange(cloud, cloud_roi);

  // Segment object using euclidean clustering
  std::vector<pcl::PointIndices> cluster_indices;
  euClustering(cloud_roi, cluster_indices);

  // B-Box & Visualize result
  sensor_msgs::msg::PointCloud2 segmented_msg;
  visualization_msgs::msg::MarkerArray markers;

  // buildSegments(cloud_roi, cluster_indices, msg->header, segmented_msg, markers);
  buildSegments(cloud_roi, cluster_indices, msg->header, segmented_msg, markers);

  cloud_pub_->publish(segmented_msg);
  marker_pub_->publish(markers);
}

void depthProcess::filterMaxRange(
  const pcl::PointCloud<PointT>::Ptr &cloud_in,
  pcl::PointCloud<PointT>::Ptr &cloud_out
) {
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(cloud_in);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, 1.8);
  pass.filter(*cloud_out);

  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(cloud_out);
  vg.setLeafSize(0.01f, 0.01f, 0.01f);
  vg.filter(*cloud_out);
}

void depthProcess::rmGroundPlane(
  const pcl::PointCloud<PointT>::Ptr &cloud_in,
  pcl::PointCloud<PointT>::Ptr &cloud_out
) {
  // RANSAC: plane removal
  pcl::SACSegmentation<PointT> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>(*cloud_in));

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(sac_distance_threshold_);
  seg.setMaxIterations(sac_max_iter_);

  seg.setInputCloud(cloud_filtered);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.empty()) {
    RCLCPP_WARN(this->get_logger(), "No ground plane found, skipping ground removal");
    *cloud_out = *cloud_filtered;
    return;
  }

  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(true);    // Remove plane inliers
  extract.filter(*cloud_out);
}

void depthProcess::euClustering(
  const pcl::PointCloud<PointT>::Ptr &cloud,
  std::vector<pcl::PointIndices> &cluster_indices
) {
  if (cloud->empty()) {
    RCLCPP_INFO(this->get_logger(), "Input cloud is empty during clustering");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Clustering on cloud with %zu points", cloud->points.size());

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
  tree->setInputCloud(cloud);

  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  RCLCPP_INFO(this->get_logger(), "Found %zu clusters", cluster_indices.size());
}

void depthProcess::buildSegments(
  const pcl::PointCloud<PointT>::Ptr &cloud,
  const std::vector<pcl::PointIndices> &cluster_indices,
  const std_msgs::msg::Header &header,
  sensor_msgs::msg::PointCloud2 &segmented_msg,
  visualization_msgs::msg::MarkerArray &marker_array
) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  int cluster_id = 0;
  marker_array.markers.clear();

  for (const auto &indices : cluster_indices) {
    if (indices.indices.size() < 5) {
      RCLCPP_DEBUG(
        this->get_logger(),"Skipping tiny cluster with %zu points", indices.indices.size()
      );
      continue;
    }
    uint8_t r = static_cast<uint8_t>(rand() % 256);
    uint8_t g = static_cast<uint8_t>(rand() % 256);
    uint8_t b = static_cast<uint8_t>(rand() % 256);

    float min_x =  std::numeric_limits<float>::max();
    float min_y =  std::numeric_limits<float>::max();
    float min_z =  std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float max_z = -std::numeric_limits<float>::max();

    for (int idx : indices.indices) {
      const auto &p = cloud->points[idx];
      pcl::PointXYZRGB cp;
      cp.x = p.x;
      cp.y = p.y;
      cp.z = p.z;
      cp.r = r;
      cp.g = g;
      cp.b = b;
      colored_cloud->points.push_back(cp);

      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      min_z = std::min(min_z, p.z);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
      max_z = std::max(max_z, p.z);
    }

    float size_x = max_x - min_x;
    float size_y = max_y - min_y;
    float size_z = max_z - min_z;

    const float minSize = 0.01f;
    if (size_x < minSize) size_x = minSize;
    if (size_y < minSize) size_y = minSize;
    if (size_z < minSize) size_z = minSize;

    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "objects";
    marker.id = cluster_id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = (min_x + max_x) / 2.0;
    marker.pose.position.y = (min_y + max_y) / 2.0;
    marker.pose.position.z = (min_z + max_z) / 2.0;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = size_x;
    marker.scale.y = size_y;
    marker.scale.z = size_z;

    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.5f;
    marker.lifetime = rclcpp::Duration(0, 1000);

    marker_array.markers.push_back(marker);
    cluster_id++;
  }

  colored_cloud->width = colored_cloud->points.size();
  colored_cloud->height = 1;
  colored_cloud->is_dense = false;

  pcl::toROSMsg(*colored_cloud, segmented_msg);
  segmented_msg.header = header;
}


depthProcess::depthProcess() : Node("depth_process") {
  this->declare_parameter<std::string>("input_cloud_topic", "/camera/camera/depth/color/points");
  this->declare_parameter<std::string>("segmented_cloud_topic", "/segmented_cloud");
  this->declare_parameter<std::string>("marker_topic", "/objects_markers");
  this->declare_parameter<double>("sac_distance_threshold", 0.05); // 5 cm
  this->declare_parameter<int>("sac_max_iter", 25000);
  this->declare_parameter<double>("cluster_tolerance", 0.05); // 5 cm
  this->declare_parameter<int>("min_cluster_size", 100);
  this->declare_parameter<int>("max_cluster_size", 25000);

  input_topic_ = this->get_parameter("input_cloud_topic").as_string();
  cloud_pub_topic_ = this->get_parameter("segmented_cloud_topic").as_string();
  marker_pub_topic_ = this->get_parameter("marker_topic").as_string();

  sac_distance_threshold_ = this->get_parameter("sac_distance_threshold").as_double();
  sac_max_iter_ = this->get_parameter("sac_max_iter").as_int();

  cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
  min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
  max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();

  RCLCPP_INFO(
    this->get_logger(),
    "cluster_tolerance = %.3f, min_cluster_size = %d, max_cluster_size = %d",
    cluster_tolerance_, min_cluster_size_, max_cluster_size_
  );

  RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", input_topic_.c_str());

  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, 10, std::bind(&depthProcess::pcCallback, this, std::placeholders::_1)
  );

  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_pub_topic_, 10);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_pub_topic_, 10);
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<depthProcess>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
