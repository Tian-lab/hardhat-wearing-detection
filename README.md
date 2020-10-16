# hardhat-wearing-detection
Construction worker hardhat-wearing detection based on an improved BiFPN

The code is Construction worker hardhat-wearing detection based on an improved BiFPN paper's test code.
To run the code, you need to do the following steps:
  1. Create checkpoint folder in the root path.
  2. Download the trained model from https://pan.baidu.com/s/15mBq-_snJBnhL6LY_EP-sA (password: sff2) and put it in the checkpoint folder.
  3. Create the dataset folder under the ./data/.
  4. Write the corresponding data of the test dataset in ./dataset/.
  Data format: image_ path x_ min,y_ min,x_ max,y_ max,class_ id ...
  5. Run evaluate.py
  6. Run mAP.py. You will get result of the mAP.
