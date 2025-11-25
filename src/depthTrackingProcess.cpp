/**
 * @author Kho Geonwoo
 */

#include <algorithm>

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

#include "interact_with_depth/depthTrackingProcess.hpp"


void depthTrackingProcess::pcCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg
) {
  pcl::PointCloud<PointT>::Ptr raw_cloud(new pcl::PointCloud<PointT>());
  pcl::fromROSMsg(*msg, *raw_cloud);

  if (raw_cloud->empty()) {
    RCLCPP_WARN(this->get_logger(), "Input PointCloud msg is EMPTY.");
    return;
  }

  pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>());
  filterCloud(raw_cloud, filtered_cloud);

  if (filtered_cloud->empty()) {
    RCLCPP_WARN(this->get_logger(), "Filtered PointCloud msg is EMPTY.");
    return;
  }

  std::vector<pcl::PointIndices> cluster_indices;
  euClustering(filtered_cloud, cluster_indices);

  RCLCPP_DEBUG(this->get_logger(), "Found %zu raw clusters", cluster_indices.size());

  std::vector<Cluster> clusters;
  computeClusters(filtered_cloud, cluster_indices, clusters);

  updateTracks(clusters, msg->header.stamp);

  buildOutput(filtered_cloud, clusters, msg->header);
}

void depthTrackingProcess::filterCloud(
  const pcl::PointCloud<PointT>::Ptr &cloud_in,
  pcl::PointCloud<PointT>::Ptr &cloud_out
) {
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(cloud_in),
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, max_range_);
  pass.filter(*cloud_out);

  RCLCPP_DEBUG(
    this->get_logger(), "Pass Through filter: %zu -> %zu points",
    cloud_in->points.size(), cloud_out->points.size()
  );

  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(cloud_out);
  vg.setLeafSize(0.01f, 0.01f, 0.01f);
  vg.filter(*cloud_out);

  RCLCPP_DEBUG(
    this->get_logger(), "Voxel Grid filter: %zu -> %zu points",
    cloud_in->points.size(), cloud_out->points.size()
  );
}

void depthTrackingProcess::euClustering(
  const pcl::PointCloud<PointT>::Ptr &cloud,
  std::vector<pcl::PointIndices> &cluster_indices
) {
  if (cloud->empty()) {
    RCLCPP_WARN(this->get_logger(), "Input cloud empty in euClustering");
    return;
  }

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
  tree->setInputCloud(cloud);

  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(max_cluster_size_);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);
}

void depthTrackingProcess::computeClusters(
  const pcl::PointCloud<PointT>::Ptr &cloud,
  const std::vector<pcl::PointIndices> &cluster_indices,
  std::vector<Cluster> &clusters
) {
  clusters.clear();
  clusters.reserve(cluster_indices.size());

  for (const auto &ci : cluster_indices) {
    if (ci.indices.size() < static_cast<size_t>(min_cluster_size_)) {
      continue;
    }

    Cluster c;
    c.indices = ci.indices;
    c.track_id = -1;

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;

    c.min_x =  std::numeric_limits<float>::max();
    c.min_y =  std::numeric_limits<float>::max();
    c.min_z =  std::numeric_limits<float>::max();
    c.max_x = -std::numeric_limits<float>::max();
    c.max_y = -std::numeric_limits<float>::max();
    c.max_z = -std::numeric_limits<float>::max();

    for (int idx : ci.indices) {
      const auto &p = cloud->points[idx];

      sum_x += p.x;
      sum_y += p.y;
      sum_z += p.z;

      c.min_x = std::min(c.min_x, p.x);
      c.min_y = std::min(c.min_y, p.y);
      c.min_z = std::min(c.min_z, p.z);
      c.max_x = std::max(c.max_x, p.x);
      c.max_y = std::max(c.max_y, p.y);
      c.max_z = std::max(c.max_z, p.z);
    }

    float inv_n = 1.0f / static_cast<float>(ci.indices.size());
    c.cx = sum_x * inv_n;
    c.cy = sum_y * inv_n;
    c.cz = sum_z * inv_n;

    clusters.push_back(c);
  }
}

void depthTrackingProcess::updateTracks(
  std::vector<Cluster> &clusters,
  const rclcpp::Time &stamp
) {
  const double assoc_dist2 = association_distance_ * association_distance_;

  std::vector<bool> track_taken(objects_.size(), false);

  std::uniform_int_distribution<int> color_dist(50, 255);

  for (auto &c : clusters) {
    int best_idx = -1;
    double best_d2 = assoc_dist2;

    for (size_t i = 0; i < objects_.size(); ++i) {
      if (track_taken[i]) continue;

      const auto &t = objects_[i];
      double dx = c.cx - t.cx;
      double dy = c.cy - t.cy;
      double dz = c.cz - t.cz;
      double d2 = dx*dx + dy*dy + dz*dz;

      if (d2 < best_d2) {
        best_d2 = d2;
        best_idx = static_cast<int>(i);
      }
    }

    if (best_idx >= 0) {
      Object &o = objects_[best_idx];
      o.cx = c.cx;
      o.cy = c.cy;
      o.cz = c.cz;
      o.last_seen = stamp;
      track_taken[best_idx] = true;

      c.track_id = o.id;
      c.r = o.r;
      c.g = o.g;
      c.b = o.b;
    } else {
      Object o;
      o.id = next_track_id_++;
      o.cx = c.cx;
      o.cy = c.cy;
      o.cz = c.cz;
      o.last_seen = stamp;
      o.r = static_cast<uint8_t>(color_dist(rng_));
      o.g = static_cast<uint8_t>(color_dist(rng_));
      o.b = static_cast<uint8_t>(color_dist(rng_));

      objects_.push_back(o);

      c.track_id = o.id;
      c.r = o.r;
      c.g = o.g;
      c.b = o.b;
    }
  }

  rclcpp::Duration timeout = rclcpp::Duration::from_seconds(track_timeout_sec_);
  objects_.erase(
    std::remove_if(
      objects_.begin(), objects_.end(),
      [&](const Object &t) {
        return (stamp - t.last_seen) > timeout;
      }),
    objects_.end());
}

void depthTrackingProcess::buildOutput(
  const pcl::PointCloud<PointT>::Ptr &cloud,
  const std::vector<Cluster> &clusters,
  const std_msgs::msg::Header &header
) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());

  visualization_msgs::msg::MarkerArray marker_array;
  marker_array.markers.clear();

  for (const auto &c : clusters) {
    // 포인트 색 입히기
    for (int idx : c.indices) {
      const auto &p = cloud->points[idx];
      pcl::PointXYZRGB cp;
      cp.x = p.x;
      cp.y = p.y;
      cp.z = p.z;
      cp.r = c.r;
      cp.g = c.g;
      cp.b = c.b;
      colored->points.push_back(cp);
    }

    // bbox Marker
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "objects";
    marker.id = c.track_id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = (c.min_x + c.max_x) / 2.0f;
    marker.pose.position.y = (c.min_y + c.max_y) / 2.0f;
    marker.pose.position.z = (c.min_z + c.max_z) / 2.0f;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    float size_x = c.max_x - c.min_x;
    float size_y = c.max_y - c.min_y;
    float size_z = c.max_z - c.min_z;

    const float kMinSize = 0.01f;
    if (size_x < kMinSize) size_x = kMinSize;
    if (size_y < kMinSize) size_y = kMinSize;
    if (size_z < kMinSize) size_z = kMinSize;

    marker.scale.x = size_x;
    marker.scale.y = size_y;
    marker.scale.z = size_z;

    marker.color.r = c.r / 255.0f;
    marker.color.g = c.g / 255.0f;
    marker.color.b = c.b / 255.0f;
    marker.color.a = 0.6f;

    marker.lifetime = rclcpp::Duration(0.0, 1000);

    marker_array.markers.push_back(marker);
  }

  colored->width = colored->points.size();
  colored->height = 1;
  colored->is_dense = false;

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*colored, cloud_msg);
  cloud_msg.header = header;

  cloud_pub_->publish(cloud_msg);
  marker_pub_->publish(marker_array);
}

depthTrackingProcess::depthTrackingProcess()
: Node("depth_tracking_process"),
  input_topic_("/camera/camera/depth/color/points"),
  cloud_pub_topic_("/tracked_cloud"),
  marker_pub_topic_("/tracked_objects"),
  max_range_(1.8),
  cluster_tolerance_(0.05),
  min_cluster_size_(50),
  max_cluster_size_(100000),
  association_distance_(0.25),
  track_timeout_sec_(1.0),
  next_track_id_(0),
  rng_(std::random_device{}()) {

  this->declare_parameter<std::string>("input_cloud_topic", input_topic_);
  this->declare_parameter<std::string>("segmented_cluod_topic", cloud_pub_topic_);
  this->declare_parameter<std::string>("marker_topic", marker_pub_topic_);
  this->declare_parameter<double>("max_range", max_range_);
  this->declare_parameter<double>("cluster_tolerance", cluster_tolerance_);
  this->declare_parameter<int>("min_cluster_size", min_cluster_size_);
  this->declare_parameter<int>("max_cluster_size", max_cluster_size_);
  this->declare_parameter<double>("association_distance", association_distance_);
  this->declare_parameter<double>("track_timeout_sec", track_timeout_sec_);

  this->get_parameter("input_cloud_topic", input_topic_);
  this->get_parameter("segmented_cluod_topic", cloud_pub_topic_);
  this->get_parameter("marker_topic", marker_pub_topic_);
  this->get_parameter("max_range", max_range_);
  this->get_parameter("cluster_tolerance", cluster_tolerance_);
  this->get_parameter("min_cluster_size", min_cluster_size_);
  this->get_parameter("max_cluster_size", max_cluster_size_);
  this->get_parameter("association_distance", association_distance_);
  this->get_parameter("track_timeout_sec", track_timeout_sec_);

  RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", input_topic_.c_str());
  RCLCPP_INFO(
    this->get_logger(),
    "cluster_tolerance=%.3f, min_cluster_size=%d, max_cluster_size=%d, "
    "assoc_dist=%.3f, max_range=%.2f",
    cluster_tolerance_, min_cluster_size_, max_cluster_size_,
    association_distance_, max_range_
  );

  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    input_topic_, 10, std::bind(&depthTrackingProcess::pcCallback, this, std::placeholders::_1)
  );

  cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_pub_topic_, 10);
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_pub_topic_, 10);
}


// main
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<depthTrackingProcess>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}