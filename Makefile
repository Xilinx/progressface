# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
all:
	cd rcnn/cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../
	cd rcnn/pycocotools/; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd rcnn/cython/; rm *.so *.c *.cpp; cd ../../
	cd rcnn/pycocotools/; rm *.so; cd ../../