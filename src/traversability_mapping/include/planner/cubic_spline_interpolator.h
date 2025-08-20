/*********************************************************************
*
* Software License Agreement (BSD License)
*
* Copyright (c) 2016, George Kouros.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above
* copyright notice, this list of conditions and the following
* disclaimer in the documentation and/or other materials provided
* with the distribution.
* * Neither the name of the the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived
* from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
* Author:  George Kouros
*********************************************************************/

#ifndef PATH_SMOOTHING_ROS_CUBIC_SPLINE_INTERPOLATOR_H
#define PATH_SMOOTHING_ROS_CUBIC_SPLINE_INTERPOLATOR_H

#include <vector>
#include <string>
#include <cmath> // For std::hypot, std::atan2, std::pow
#include <algorithm> // For std::min, std::max

// ROS 2 相关的头文件
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h" // For tf2::Quaternion
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // For tf2::toMsg
#include "tf2/utils.h" // For tf2::getYaw

// 如果这个类将作为独立节点的一部分，可能需要rclcpp
// 但如果它只是一个库，则不需要直接包含rclcpp。
// 考虑到构造函数中对参数服务器的访问，这里假定它会与rclcpp::Node一起使用。
#include "rclcpp/rclcpp.hpp" // 用于参数访问


namespace path_smoothing
{

  class CubicSplineInterpolator
  {
    public:

      // 新增一个接受 rclcpp::Node::SharedPtr 的构造函数，用于参数访问
      explicit CubicSplineInterpolator(
        rclcpp::Node::SharedPtr node_ptr,
        double pointsPerUnit = 5.0,
        unsigned int skipPoints = 0,
        bool useEndConditions = true,
        bool useMiddleConditions = false);

      // 旧的构造函数，用于没有ROS参数服务器依赖的场景
      CubicSplineInterpolator(
        double pointsPerUnit = 5.0,
        unsigned int skipPoints = 0,
        bool useEndConditions = true,
        bool useMiddleConditions = false);

      // 用于从参数服务器加载参数的构造函数
      explicit CubicSplineInterpolator(rclcpp::Node::SharedPtr node_ptr, std::string name);

      ~CubicSplineInterpolator();

      void interpolatePath(
        const nav_msgs::msg::Path& path, nav_msgs::msg::Path& smoothedPath);

      void interpolatePath(
        const std::vector<geometry_msgs::msg::PoseStamped>& path,
        std::vector<geometry_msgs::msg::PoseStamped>& smoothedPath);

      int interpolatePoint(
        const std::vector<geometry_msgs::msg::PoseStamped>& path,
        const std::vector<double>& cummulativeDistances,
        geometry_msgs::msg::PoseStamped& point,
        double pointCummDist);

      void calcCummulativeDistances(
        const std::vector<geometry_msgs::msg::PoseStamped> path,
        std::vector<double>& cummulativeDistances);

      double calcTotalDistance(const std::vector<geometry_msgs::msg::PoseStamped>& path);

      double calcDistance(
        const std::vector<geometry_msgs::msg::PoseStamped>& path,
        unsigned int idx);

      double calcAlphaCoeff(
        const std::vector<geometry_msgs::msg::PoseStamped> path,
        const std::vector<double> cummulativeDistances,
        unsigned int idx,
        double input);

      double calcBetaCoeff(
        const std::vector<geometry_msgs::msg::PoseStamped> path,
        const std::vector<double> cummulativeDistances,
        unsigned int idx,
        double input);

      double calcGammaCoeff(
        const std::vector<geometry_msgs::msg::PoseStamped> path,
        const std::vector<double> cummulativeDistances,
        unsigned int idx,
        double input);

      double calcDeltaCoeff(
        const std::vector<geometry_msgs::msg::PoseStamped> path,
        const std::vector<double> cummulativeDistances,
        unsigned int idx,
        double input);

      double calcRelativeDistance(
        const std::vector<double>& cummulativeDistances,
        unsigned int idx,
        double input);

      void calcPointGradient(
        const std::vector<geometry_msgs::msg::PoseStamped>& path,
        const std::vector<double>& cummulativeDistances,
        unsigned int idx, std::vector<double>& gradient);

      unsigned int findGroup(
        const std::vector<double>& cummulativeDistances,
        double pointCummDist);

      double getPointsPerUnit() {return pointsPerUnit_;}
      unsigned int skipPoints() {return skipPoints_;}
      bool getUseEndConditions() {return useEndConditions_;}
      bool getUseMiddleConditions() {return useMiddleConditions_;}

      void setPointsPerUnit(double ppu) {pointsPerUnit_ = ppu;}
      void setSkipPoints(unsigned int sp) {skipPoints_ = sp;}
      void setUseEndConditions(bool uec) {useEndConditions_ = uec;}
      void setUseMiddleConditions(bool umc) {useMiddleConditions_ = umc;}

    private:
      double pointsPerUnit_;
      unsigned int skipPoints_;
      bool useEndConditions_;
      bool useMiddleConditions_;
      rclcpp::Node::SharedPtr node_ptr_; // 用于ROS 2参数访问的节点指针
  };

}  // namespace path_smoothing

#endif  // PATH_SMOOTHING_ROS_CUBIC_SPLINE_INTERPOLATOR_H




namespace path_smoothing
{

  // 默认构造函数（无ROS参数访问）
  CubicSplineInterpolator::CubicSplineInterpolator(
    double pointsPerUnit,
    unsigned int skipPoints,
    bool useEndConditions,
    bool useMiddleConditions)
    :
      pointsPerUnit_(pointsPerUnit),
      skipPoints_(skipPoints),
      useEndConditions_(useEndConditions),
      useMiddleConditions_(useMiddleConditions),
      node_ptr_(nullptr) // 初始化为nullptr
  {
  }

  // 接收rclcpp::Node::SharedPtr的构造函数
  CubicSplineInterpolator::CubicSplineInterpolator(
    rclcpp::Node::SharedPtr node_ptr,
    double pointsPerUnit,
    unsigned int skipPoints,
    bool useEndConditions,
    bool useMiddleConditions)
    :
      pointsPerUnit_(pointsPerUnit),
      skipPoints_(skipPoints),
      useEndConditions_(useEndConditions),
      useMiddleConditions_(useMiddleConditions),
      node_ptr_(node_ptr) // 存储节点指针
  {
  }


  // 从参数服务器加载参数的构造函数
  CubicSplineInterpolator::CubicSplineInterpolator(rclcpp::Node::SharedPtr node_ptr, std::string name)
    : node_ptr_(node_ptr)
  {
    // 声明参数
    node_ptr_->declare_parameter<double>(name + ".points_per_unit", 5.0);
    node_ptr_->declare_parameter<bool>(name + ".use_end_conditions", false);
    node_ptr_->declare_parameter<bool>(name + ".use_middle_conditions", false);
    node_ptr_->declare_parameter<int>(name + ".skip_points", 0);

    // 获取参数
    node_ptr_->get_parameter(name + ".points_per_unit", pointsPerUnit_);
    node_ptr_->get_parameter(name + ".use_end_conditions", useEndConditions_);
    node_ptr_->get_parameter(name + ".use_middle_conditions", useMiddleConditions_);

    int skipPoints_int;
    node_ptr_->get_parameter(name + ".skip_points", skipPoints_int);
    skipPoints_ = static_cast<unsigned int>(std::abs(skipPoints_int));

    // 可以添加RCLCPP_INFO来确认参数加载
    RCLCPP_INFO(node_ptr_->get_logger(), "CubicSplineInterpolator initialized with parameters:");
    RCLCPP_INFO(node_ptr_->get_logger(), "  points_per_unit: %f", pointsPerUnit_);
    RCLCPP_INFO(node_ptr_->get_logger(), "  use_end_conditions: %d", useEndConditions_);
    RCLCPP_INFO(node_ptr_->get_logger(), "  use_middle_conditions: %d", useMiddleConditions_);
    RCLCPP_INFO(node_ptr_->get_logger(), "  skip_points: %u", skipPoints_);
  }


  CubicSplineInterpolator::~CubicSplineInterpolator()
  {
  }


  void CubicSplineInterpolator::interpolatePath(
    const nav_msgs::msg::Path& path,
    nav_msgs::msg::Path& smoothedPath)
  {
    smoothedPath.header = path.header;
    interpolatePath(path.poses, smoothedPath.poses);
  }


  void CubicSplineInterpolator::interpolatePath(
    const std::vector<geometry_msgs::msg::PoseStamped>& path,
    std::vector<geometry_msgs::msg::PoseStamped>& smoothedPath)
  {
    // clear new smoothed path vector in case it's not empty
    smoothedPath.clear();

    // set skipPoints_ to 0 if the path contains has too few points
    unsigned int oldSkipPoints = skipPoints_;
    // 确保 path.size() 足够大，避免负值或越界
    skipPoints_ = std::min(static_cast<unsigned int>(path.size() > 2 ? path.size() - 2 : 0), skipPoints_);

    // create cummulative distances vector
    std::vector<double> cummulativeDistances;
    calcCummulativeDistances(path, cummulativeDistances);  //计算每个采样点到起点的距离

    // create temp pose
    geometry_msgs::msg::PoseStamped pose;
    if (!path.empty()) {
      pose.header = path[0].header;
    } else {
      // 如果路径为空，设置一个默认头，或者抛出错误
      RCLCPP_WARN(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "Input path is empty in interpolatePath.");
      return;
    }

    /*pointsPerUnit_ 是一个类成员变量,表示每个单位距离上需要插值的点数。
    calcTotalDistance(path) 是一个辅助函数,用于计算输入路径 path 的总长度。
    将总长度乘以每个单位距离上需要的点数,得到平滑后路径需要的总点数 numPoints。*/
    unsigned int numPoints = static_cast<unsigned int>(pointsPerUnit_ * calcTotalDistance(path));
    if (numPoints == 0) {
        RCLCPP_WARN(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "Calculated numPoints is 0. No interpolation performed.");
        smoothedPath.clear();
        return;
    }

    smoothedPath.resize(numPoints);

    double groupUStart = 0;
    int groupID = -1;
    // interpolate points on the smoothed path using the points in the original path
    for (unsigned int i = 0; i < numPoints; i++)
    {
      double u = static_cast<double>(i) / (numPoints-1);
      int group = interpolatePoint(path, cummulativeDistances, pose, u);

      if (groupID != group){
        groupID = group;
        groupUStart = u;
      }

      // Check for NaN values from interpolation
      if (std::isnan(pose.pose.position.x) || std::isnan(pose.pose.position.y)) {
        if (i > 0) {
            pose.pose = smoothedPath[i-1].pose; // Use previous valid pose
        } else {
            // If the very first point is NaN, try to use the first point from the original path
            if (!path.empty()) {
                pose.pose = path[0].pose;
            } else {
                // Fallback if original path is also empty (should have been caught earlier)
                RCLCPP_ERROR(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "First interpolated point is NaN and original path is empty.");
                break; // Exit loop if cannot recover
            }
        }
      }

      // Ensure cummulativeDistances[group+1] - cummulativeDistances[group] is not zero to prevent division by zero
      double thisLength = cummulativeDistances[group+1] - cummulativeDistances[group];
      if (std::abs(thisLength) > 1e-9) { // Using a small epsilon to check for non-zero
        pose.pose.position.z = (u - groupUStart) / thisLength * (path[group+1].pose.position.z - path[group].pose.position.z) + path[group].pose.position.z;
      } else {
        pose.pose.position.z = path[group].pose.position.z; // If length is zero, Z remains same as current group point
        RCLCPP_WARN_THROTTLE(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"),
                             *node_ptr_->get_clock(), 1000, "Segment length is zero at group %d. Z position might be inaccurate.", group);
      }

      smoothedPath[i] = pose;
    }

    // interpolate orientations of intermediate poses
    for (unsigned int i = 1; i < smoothedPath.size()-1; i++)
    {
      double dx = smoothedPath[i+1].pose.position.x - smoothedPath[i].pose.position.x;
      double dy = smoothedPath[i+1].pose.position.y - smoothedPath[i].pose.position.y;
      double th = atan2(dy, dx);

      tf2::Quaternion q;
      q.setRPY(0, 0, th); // Roll, Pitch, Yaw
      smoothedPath[i].pose.orientation = tf2::toMsg(q);
    }

    // revert skipPoints to original value
    skipPoints_ = oldSkipPoints;

    // the last element can be nan sometimes
    if (numPoints > 1) { // Ensure there's at least two points to avoid resizing to 0 if numPoints is 1
        smoothedPath.resize(numPoints-1);
    } else {
        smoothedPath.clear(); // If numPoints is 0 or 1, clear it or handle as needed
    }


    if (!smoothedPath.empty()) {
        if (smoothedPath.size() > 1) {
            // smoothedPath.front().pose.orientation = smoothedPath[1].pose.orientation;
            // A more robust way to set start/end orientations from original path,
            // or based on the first meaningful segment.
            // For now, mirroring original behavior.
            smoothedPath.front().pose.orientation = smoothedPath[1].pose.orientation;
        } else {
            // If only one point after resize, orientation might be missing.
            // Copy from original path if available.
            if (!path.empty()) {
                smoothedPath.front().pose.orientation = path.front().pose.orientation;
            }
        }
    }
  }


  int CubicSplineInterpolator::interpolatePoint(
    const std::vector<geometry_msgs::msg::PoseStamped>& path,
    const std::vector<double>& cummulativeDistances,
    geometry_msgs::msg::PoseStamped& point,
    double pointCummDist)
  {
    int group = findGroup(cummulativeDistances, pointCummDist);
    // RCLCPP_INFO(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "u: %f, idx: %u", pointCummDist, group);

    double a = calcAlphaCoeff(path, cummulativeDistances, group, pointCummDist);
    double b = calcBetaCoeff(path, cummulativeDistances, group, pointCummDist);
    double c = calcGammaCoeff(path, cummulativeDistances, group, pointCummDist);
    double d = calcDeltaCoeff(path, cummulativeDistances, group, pointCummDist);

    std::vector<double> grad, nextGrad;
    calcPointGradient(path, cummulativeDistances, group, grad);
    calcPointGradient(path, cummulativeDistances, group+1, nextGrad);

    // Ensure indices are within bounds before accessing path
    unsigned int current_idx = group * (skipPoints_ + 1);
    unsigned int next_idx = (group + 1) * (skipPoints_ + 1);

    if (current_idx >= path.size() || next_idx >= path.size()) {
        RCLCPP_ERROR(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "Interpolation index out of bounds. Group: %d, Path Size: %zu", group, path.size());
        // Set point to NaN or some default to indicate error
        point.pose.position.x = std::nan("");
        point.pose.position.y = std::nan("");
        return group; // Or throw an exception
    }


    point.pose.position.x =
      + a * path[current_idx].pose.position.x
      + b * path[next_idx].pose.position.x
      + c * grad[0]
      + d * nextGrad[0];

    point.pose.position.y =
      + a * path[current_idx].pose.position.y
      + b * path[next_idx].pose.position.y
      + c * grad[1]
      + d * nextGrad[1];

    return group;
  }


  void CubicSplineInterpolator::calcCummulativeDistances(
    const std::vector<geometry_msgs::msg::PoseStamped> path,
    std::vector<double>& cummulativeDistances)
  {
    cummulativeDistances.clear();
    cummulativeDistances.push_back(0);

    // Ensure path has enough points to avoid out-of-bounds access
    if (path.size() <= skipPoints_ + 1) {
        if (path.size() > 0) {
            cummulativeDistances.push_back(1.0); // For very short paths, make it span 0-1
        }
        return;
    }


    double totalPathDistance = calcTotalDistance(path);
    if (totalPathDistance < 1e-9) { // Avoid division by zero if total distance is very small
        if (path.size() > 1) {
            for (unsigned int i = 1; i < path.size(); ++i) {
                cummulativeDistances.push_back(static_cast<double>(i) / (path.size() - 1));
            }
        }
        return;
    }

    for (unsigned int i = skipPoints_+1; i < path.size(); i += skipPoints_+1)
    {
        // Ensure index for calcDistance is valid
        if (i < path.size()) {
            cummulativeDistances.push_back(
                cummulativeDistances.back()
                + calcDistance(path, i) / totalPathDistance);
        } else {
             RCLCPP_WARN_THROTTLE(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"),
                                 *node_ptr_->get_clock(), 1000, "Index out of bounds in calcCummulativeDistances loop. i: %u, path size: %zu", i, path.size());
            break; // Exit loop if index is invalid
        }
    }
    // Ensure the last cumulative distance is 1.0, even if precision issues occur
    if (!cummulativeDistances.empty()) {
        cummulativeDistances.back() = 1.0;
    }
  }


  double CubicSplineInterpolator::calcTotalDistance(
    const std::vector<geometry_msgs::msg::PoseStamped>& path)
  {
    double totalDist = 0;

    // Ensure path has enough points
    if (path.size() <= skipPoints_ + 1) {
        return 0; // Not enough points to calculate segments
    }

    for (unsigned int i = skipPoints_+1; i < path.size(); i += skipPoints_+1) {
      if (i < path.size()) { // Bounds check
        totalDist += calcDistance(path, i);
      } else {
        RCLCPP_WARN_THROTTLE(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"),
                             *node_ptr_->get_clock(), 1000, "Index out of bounds in calcTotalDistance loop. i: %u, path size: %zu", i, path.size());
        break;
      }
    }

    return totalDist;
  }


  double CubicSplineInterpolator::calcDistance(
    const std::vector<geometry_msgs::msg::PoseStamped>& path,
    unsigned int idx)
  {
    // Adjusted bounds check: idx-skipPoints_-1 must be >= 0
    if (idx == 0 || idx >= path.size() || (idx < skipPoints_ + 1 && skipPoints_ + 1 != 0))
      return 0;

    double dist =
      std::hypot(
        path[idx].pose.position.x - path[idx-skipPoints_-1].pose.position.x,
        path[idx].pose.position.y - path[idx-skipPoints_-1].pose.position.y);

    return dist;
  }


  double CubicSplineInterpolator::calcAlphaCoeff(
    const std::vector<geometry_msgs::msg::PoseStamped> path, // path not strictly needed here
    const std::vector<double> cummulativeDistances,
    unsigned int idx,
    double input)
  {
    double relDist = calcRelativeDistance(cummulativeDistances, idx, input);
    double alpha =
      + 2 * std::pow(relDist, 3)
      - 3 * std::pow(relDist, 2)
      + 1;

    return alpha;
  }


  double CubicSplineInterpolator::calcBetaCoeff(
    const std::vector<geometry_msgs::msg::PoseStamped> path, // path not strictly needed here
    const std::vector<double> cummulativeDistances,
    unsigned int idx,
    double input)
  {
    double relDist = calcRelativeDistance(cummulativeDistances, idx, input);
    double beta =
      - 2 * std::pow(relDist, 3)
      + 3 * std::pow(relDist, 2);

    return beta;
  }


  double CubicSplineInterpolator::calcGammaCoeff(
    const std::vector<geometry_msgs::msg::PoseStamped> path, // path not strictly needed here
    const std::vector<double> cummulativeDistances,
    unsigned int idx,
    double input)
  {
    double relDist = calcRelativeDistance(cummulativeDistances, idx, input);
    double gamma =
      (std::pow(relDist, 3)
       - 2 * std::pow(relDist, 2))
      * (cummulativeDistances[idx+1] - cummulativeDistances[idx])
      + input
      - cummulativeDistances[idx];

    return gamma;
  }


  double CubicSplineInterpolator::calcDeltaCoeff(
    const std::vector<geometry_msgs::msg::PoseStamped> path, // path not strictly needed here
    const std::vector<double> cummulativeDistances,
    unsigned int idx,
    double input)
  {
    double relDist = calcRelativeDistance(cummulativeDistances, idx, input);
    double delta =
      (std::pow(relDist, 3)
       - std::pow(relDist, 2))
      * (cummulativeDistances[idx+1] - cummulativeDistances[idx]);

      return delta;
  }


  double CubicSplineInterpolator::calcRelativeDistance(
    const std::vector<double>& cummulativeDistances,
    const unsigned int idx,
    const double input)
  {
    // Bounds check for cummulativeDistances
    if (idx >= cummulativeDistances.size() || (idx + 1) >= cummulativeDistances.size()) {
        RCLCPP_ERROR(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "Cumulative distance index out of bounds: idx %u, size %zu", idx, cummulativeDistances.size());
        return 0.0; // Or throw an error, depending on desired behavior
    }

    double denom = cummulativeDistances[idx+1] - cummulativeDistances[idx];
    if (std::abs(denom) < 1e-9) { // Avoid division by zero
        // If segment length is zero, relative distance is undefined.
        // Return 0 or 1 based on input position relative to current point
        return (input <= cummulativeDistances[idx]) ? 0.0 : 1.0;
    }

    double relDist = (input - cummulativeDistances[idx]) / denom;
    return relDist;
  }


  void CubicSplineInterpolator::calcPointGradient(
    const std::vector<geometry_msgs::msg::PoseStamped>& path,
    const std::vector<double>& cummulativeDistances,
    unsigned int idx,
    std::vector<double>& gradient)
  {
    double dx, dy, du;
    gradient.assign(2, 0.0); // Initialize with 0.0

    // Ensure path and cummulativeDistances have enough points for the given index
    unsigned int path_idx = idx * (skipPoints_ + 1);
    if (path_idx >= path.size()) {
        RCLCPP_WARN(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "Path index out of bounds in calcPointGradient: %u (from idx %u)", path_idx, idx);
        return;
    }
    if (idx >= cummulativeDistances.size() || (idx + 1) >= cummulativeDistances.size()) {
         RCLCPP_WARN(node_ptr_ ? node_ptr_->get_logger() : rclcpp::get_logger("CubicSplineInterpolator"), "Cumulative distance index out of bounds in calcPointGradient: %u", idx);
         return;
    }


    // use either pose.yaw or interpolation to find gradient of points
    if ((useEndConditions_ && (idx == 0 || idx == cummulativeDistances.size()-1))
      || useMiddleConditions_)
    {
      double th = tf2::getYaw(path[path_idx].pose.orientation); // Use tf2::getYaw
      // int sign = (fabs(th) < M_PI / 2) ? 1 : -1; // Original logic for sign - consider if still relevant
      // A more direct calculation for gradient based on orientation
      // For a unit vector, dx = cos(th), dy = sin(th)
      // The magnitude of the gradient is the total distance scale
      double totalDist = calcTotalDistance(path);
      // Need to be careful here: gradient should be dX/du, dY/du.
      // If `th` is the heading, then the tangent vector components are cos(th) and sin(th).
      // This part of the original code might need a deeper look to ensure it calculates dX/du and dY/du correctly.
      // The original calculation: sqrt(1 + pow(tan(th),2)) / (1 + pow(tan(th), 2)) simplifies to cos(th).
      // So gradient[0] = sign * totalDist * cos(th) and gradient[1] = tan(th) * gradient[0] = sin(th) * sign * totalDist
      // This seems to scale the tangent vector by totalDist.
      gradient[0] = totalDist * std::cos(th);
      gradient[1] = totalDist * std::sin(th);

    }
    else  // gradient interpolation using original points
    {
      // Ensure indices are valid
      if (idx == 0 || idx >= cummulativeDistances.size() || (idx - 1) * (skipPoints_ + 1) >= path.size())
        return; // Not enough points for interpolation

      unsigned int prev_path_idx = (idx - 1) * (skipPoints_ + 1);
      // Ensure current_path_idx is also valid, though checked above
      // unsigned int current_path_idx = idx * (skipPoints_ + 1);

      dx = path[path_idx].pose.position.x - path[prev_path_idx].pose.position.x;
      dy = path[path_idx].pose.position.y - path[prev_path_idx].pose.position.y;
      du = cummulativeDistances[idx] - cummulativeDistances[idx-1];

      if (std::abs(du) > 1e-9) { // Avoid division by zero
        gradient[0] =  dx / du;
        gradient[1] =  dy / du;
      } else {
        // If du is zero, segment is collapsed, gradient is undefined or zero.
        gradient[0] = 0.0;
        gradient[1] = 0.0;
      }
    }
  }


  unsigned int CubicSplineInterpolator::findGroup(
    const std::vector<double>& cummulativeDistances,
    double pointCummDist)
  {
    // Handle edge cases for empty or single-point paths
    if (cummulativeDistances.empty()) {
        return 0;
    }
    if (cummulativeDistances.size() == 1) {
        return 0; // Only one group
    }

    unsigned int i;
    for (i = 0; i < cummulativeDistances.size()-1; i++)
    {
      if (pointCummDist <= cummulativeDistances[i+1])
        return i;
    }
    // If pointCummDist is greater than all values, it belongs to the last group
    return i-1; // Return the last valid group index
  }

}  // namespace path_smoothing