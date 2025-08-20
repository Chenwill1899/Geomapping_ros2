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
#ifndef _NODE_H_
#define _NODE_H_

// 移除 <ros/ros.h>
#include <Eigen/Eigen>
#include <utility>      // For std::pair, std::move etc. (though not directly used in this snippet)
#include <vector>       // For std::vector
#include <list>         // For std::list
#include <limits>       // For DBL_MAX, modern alternative to <cfloat>

// 确保 Eigen 相关的对齐宏在类定义之前
// 如果在某个函数中使用到 Eigen::Vector2d 等，并且将其存储在标准容器中
// 且该容器本身没有 Eigen::aligned_allocator，可能导致对齐问题。
// 对于 Eigen 类型作为结构体成员，EIGEN_MAKE_ALIGNED_OPERATOR_NEW 是必要的。

struct TreeNode
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 确保Eigen类型成员正确对齐
    TreeNode() : parent(NULL), cost_from_start(std::numeric_limits<double>::max()), cost_from_parent(0.0), dist_from_start(std::numeric_limits<double>::max()), dist_from_parent(0.0){};
    TreeNode *parent;
    Eigen::Vector2d x;
    double cost_from_start;
    double cost_from_parent;
    double dist_from_start;
    double dist_from_parent;
    double heuristic_to_goal;
    double g_plus_h;
    std::list<TreeNode *> children;
};

typedef TreeNode *RRTNode2DPtr;
// 明确使用 std::vector
typedef std::vector<RRTNode2DPtr, Eigen::aligned_allocator<RRTNode2DPtr>> RRTNode2DPtrVector;
typedef std::vector<TreeNode, Eigen::aligned_allocator<TreeNode>> RRTNode2DVector;

class RRTNodeComparator
{
public:
    bool operator()(RRTNode2DPtr node1, RRTNode2DPtr node2)
    {
        return node1->g_plus_h > node2->g_plus_h;
    }
};

struct NodeWithStatus
{
    NodeWithStatus()
    {
        node_ptr = nullptr;
        is_checked = false;
        is_valid = false;
    };
    NodeWithStatus(const RRTNode2DPtr &n, bool checked, bool valid) : node_ptr(n), is_checked(checked), is_valid(valid){};
    RRTNode2DPtr node_ptr;
    bool is_checked;
    bool is_valid; // the segment from a center, not only the node
};

struct Neighbour
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 如果 Eigen::Vector2d center 是成员，建议也加上
    Eigen::Vector2d center;
    // 明确使用 std::vector
    std::vector<NodeWithStatus, Eigen::aligned_allocator<NodeWithStatus>> nearing_nodes; // 如果 NodeWithStatus 包含 Eigen 类型，需要aligned_allocator
};

#endif // _NODE_H_