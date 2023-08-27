// MIT License

// Copyright (c) 2023  Miguel Ángel González Santamarta

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef LLAMA_ROS__LLAMA_NODE_HPP
#define LLAMA_ROS__LLAMA_NODE_HPP

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>

#include <memory>
#include <string>

#include "llama.h"
#include "llama_msgs/GenerateResponseAction.h"
#include "llama_msgs/GenerateEmbeddings.h"
#include "llama_msgs/Tokenize.h"
#include "llama_ros/llama.hpp"
namespace llama_ros {

class LlamaNode {

public:
  LlamaNode();

private:
  std::shared_ptr<Llama> llama;
  
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  bool use_default_sampling_config_;

  // ros1
  ros::ServiceServer tokenize_service_;
  ros::ServiceServer generate_embeddings_service_;
  std::unique_ptr<
      actionlib::SimpleActionServer<llama_msgs::GenerateResponseAction>>
      generate_response_action_server_;
  std::mutex handle_accepted_mtx_;

  // methods
  bool tokenize_service_callback(
      llama_msgs::Tokenize::Request &request,
      llama_msgs::Tokenize::Response &response);
  bool generate_embeddings_service_callback(
     llama_msgs::GenerateEmbeddings::Request &request,
     llama_msgs::GenerateEmbeddings::Response &response);

  void execute(const llama_msgs::GenerateResponseGoalConstPtr &goal);
  void send_text(const completion_output &completion);
};

} // namespace llama_ros

#endif
