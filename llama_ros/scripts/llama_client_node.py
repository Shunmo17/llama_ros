#!/usr/bin/env python3

# MIT License

# Copyright (c) 2023  Miguel Ángel González Santamarta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import rospy
import actionlib
from llama_msgs.msg import GenerateResponseAction, GenerateResponseGoal, GenerateResponseFeedback


class LlamaClientNode():

    def __init__(self) -> None:
        self.prompt = rospy.get_param("~prompt")
        self.prompt = self.prompt.replace("\\n", "\n")

        self._get_result_future = None
        self._action_client = actionlib.SimpleActionClient("/generate_response",
            GenerateResponseAction)

    def text_cb(self, msg) -> None:
        feedback: GenerateResponseFeedback = msg
        rospy.logdebug(feedback.partial_response.text)

    def send_prompt(self) -> None:

        goal = GenerateResponseGoal()
        goal.prompt = self.prompt
        goal.sampling_config.temp = 0.2
        goal.sampling_config.repeat_last_n = 8

        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal(
            goal, feedback_cb=self.text_cb)
        self._action_client.wait_for_result()

        result: GenerateResponse.Result = self._action_client.get_result()

        print("//////////////////////////////////////////////////")
        rospy.loginfo(len(result.response.text.split(r"\n")))
        for txt in result.response.text.split(r"\n"):
            rospy.loginfo(txt)


def main():

    rospy.init_node("llama_client_node")
    LlamaClientNode().send_prompt()


if __name__ == "__main__":
    main()
