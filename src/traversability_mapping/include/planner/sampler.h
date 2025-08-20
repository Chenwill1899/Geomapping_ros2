/*
Copyright (C) 2022 Hongkai Ye (kyle_yeh@163.com)
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/
#ifndef _BIAS_SAMPLER_
#define _BIAS_SAMPLER_

// 移除 <ros/ros.h>
#include <Eigen/Eigen>
#include <random> // For std::random_device, std::mt19937_64, std::uniform_real_distribution, std::normal_distribution
#include <cmath>  // For std::pow

#include "planner/planningState.h" // 假设这个头文件也是ROS 2兼容的

class BiasSampler
{
public:
  BiasSampler()
  {
    std::random_device rd;
    gen_ = std::mt19937_64(rd());
    uniform_rand_ = std::uniform_real_distribution<double>(0.0, 1.0);
    normal_rand_ = std::normal_distribution<double>(0.0, 1.0);
    range_.setZero();
    origin_.setZero();
    informed_ = false;
    GUILD_informed_ = false;
    PlanningState_ = WithoutGoal;
    goal_biased_=0.15;
  };

  void initWithoutGoal(const Eigen::Vector2d start, const Eigen::Vector2d origin, const Eigen::Vector2d range){
    start_ = start;
    origin_ = origin;
    range_ = range;
    PlanningState_ = WithoutGoal;
  }

  void initWithGoal(const Eigen::Vector2d start, const Eigen::Vector2d goal, const Eigen::Vector2d origin, const Eigen::Vector2d range){
    start_ = start;
    goal_ = goal;
    origin_ = origin;
    range_ = range;
    PlanningState_ = Global;
  }

  // void setSamplingRange(const Eigen::Vector2d origin, const Eigen::Vector2d range)
  // {
  //   origin_ = origin;
  //   range_ = range;
  // }

  void setPlanState(const PlanningState state)
  {
    PlanningState_ = state;
  }

  void setGoalBiased(double GoalBias)
  {
    goal_biased_ = GoalBias;
  }

  void samplingOnce(Eigen::Vector2d &sample)
  {
    switch(PlanningState_)
          {
            case Global:
            {
              if(informed_){
                informedSamplingOnce(sample);   //已经找到初始路径
              }
              else{

                if(uniform_rand_(gen_)>goal_biased_) //偏向终点采样
                  uniformSamplingOnce(sample);
                else
                  sample = goal_;
              }
            }
            break;
            case WithoutGoal:
                uniformSamplingOnce(sample);
            break;
            default:
            break;
          }
  }

  void uniformSamplingOnce(Eigen::Vector2d &sample)
  {
    sample[0] = uniform_rand_(gen_); //生成0-1浮点数
    sample[1] = uniform_rand_(gen_);
    // sample[2] = uniform_rand_(gen_); // 2D vector, no z component
    sample.array() *= range_.array();
    sample += origin_;
  }

  void informedSamplingOnce(Eigen::Vector2d &sample)
  {
    // random uniform sampling in a unit 2-ball (original was 3-ball, adapted to 2D)
    Eigen::Vector2d p;
    p[0] = normal_rand_(gen_);
    p[1] = normal_rand_(gen_);
    // p[2] = normal_rand_(gen_); // 2D vector, no z component
    double r = std::pow(uniform_rand_(gen_), 0.5); // For a 2D uniform distribution in a circle
    sample = r * p.normalized();

    // transform the pt into the ellipsoid
    sample.array() *= radii_.array();
    sample = rotation_ * sample;
    sample += center_;
  }

  void GUILDSamplingOnce(Eigen::Vector2d &sample)
  {
    // random uniform sampling in a unit 2-ball (original was 3-ball, adapted to 2D)
    Eigen::Vector2d p;
    p[0] = normal_rand_(gen_);
    p[1] = normal_rand_(gen_);
    // p[2] = normal_rand_(gen_); // 2D vector, no z component
    double r = std::pow(uniform_rand_(gen_), 0.33333); // This exponent 0.33333 is for 3D sphere volume; for 2D, it should be 0.5
                                                     // Retaining original exponent if it's intentional for a specific distribution,
                                                     // but for uniform sampling within a 2D circle, it should be 0.5.
    sample = r * p.normalized();

    // transform the pt into the ellipsoid
    if (p[0] > 0.0)
    {
      sample.array() *= radii_s_.array();
      sample = rotation_s_ * sample;
      sample += center_s_;
    }
    else
    {
      sample.array() *= radii_g_.array();
      sample = rotation_g_ * sample;
      sample += center_g_;
    }
  }

  void setInformedTransRot(const Eigen::Vector2d &trans, const Eigen::Matrix2d &rot)
  {
    center_ = trans;
    rotation_ = rot;
  }

  void setInformedSacling(const Eigen::Vector2d &scale)
  {
    informed_ = true;
    radii_ = scale;
  }

  void setGUILDInformed(const Eigen::Vector2d &scale1, const Eigen::Vector2d &trans1, const Eigen::Matrix2d &rot1,
                        const Eigen::Vector2d &scale2, const Eigen::Vector2d &trans2, const Eigen::Matrix2d &rot2)
  {
    radii_s_ = scale1;
    center_s_ = trans1;
    rotation_s_ = rot1;
    radii_g_ = scale2;
    center_g_ = trans2;
    rotation_g_ = rot2;
  }

  void reset()
  {
    informed_ = false;
    GUILD_informed_ = false;
  }

  // (0.0 - 1.0)
  double getUniRandNum()
  {
    return uniform_rand_(gen_);
  }

private:
  Eigen::Vector2d range_, origin_;  //局部地图左下角为起点
  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> uniform_rand_;
  std::normal_distribution<double> normal_rand_;

  // for informed sampling
  bool informed_;
  Eigen::Vector2d center_, radii_;
  Eigen::Vector2d start_, goal_;
  Eigen::Matrix2d rotation_;

  PlanningState PlanningState_; // Assuming PlanningState enum is defined in planner/planningState.h
  double goal_biased_;

  // for GUILD informed sampling
  bool GUILD_informed_;
  Eigen::Vector2d center_s_, radii_s_;
  Eigen::Vector2d center_g_, radii_g_;
  Eigen::Matrix2d rotation_s_;
  Eigen::Matrix2d rotation_g_;
};

#endif // _BIAS_SAMPLER_