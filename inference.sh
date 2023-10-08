# activate virtual env
#conda activate HiSIF3
#cd source directory
cd /media/HiSIF-DTA

echo "Start Inferencing ..."
echo "##################################################################################################"
echo "\n"

echo "********************************************************"
echo "The test of the model's performance on the DTA task is about to commence. Waiting...."
wait
echo "********************************************************"
echo "Test the performance of the Bottom-Up model on the davis dataset"
python test_for_DTA.py --model BUNet --dataset davis
wait
echo "********************************************************"
echo "Test the performance of the Top-Down model on the davis dataset"
python test_for_DTA.py --model TDNet --dataset davis
wait
echo "********************************************************"
echo "Test the performance of the Bottom-Up model on the kiba dataset"
python test_for_DTA.py --model BUNet --dataset kiba
wait
echo "********************************************************"
echo "Test the performance of the Top-Down model on the kiba dataset"
python test_for_DTA.py --model TDNet --dataset kiba
wait
echo "********************************************************"
echo "\n"

echo "********************************************************"
echo "The test of the model's performance on the CPI task is about to commence. Waiting...."
wait
echo "********************************************************"
echo "Test the performance of the Bottom-Up model on the Human dataset (5-fold) "
python test_for_CPI.py --model BUNet --dataset Human
wait
echo "********************************************************"
echo "Test the performance of the Top-Down model on the Human dataset (5-fold)"
python test_for_CPI.py --model TDNet --dataset Human
wait
echo "********************************************************"
echo "\n"

echo "********************************************************"
echo "The performance of various ablation model variants is about to be presented. Waiting...."
wait
echo "********************************************************"
echo "Test the performance of the LSM variant model on the davis dataset "
python test_for_Ablation.py --model LSM --dataset davis
wait
echo "********************************************************"
echo "Test the performance of the HSM variant model on the davis dataset"
python test_for_Ablation.py --model HSM --dataset davis
wait
echo "********************************************************"
echo "Test the performance of the HFS variant model on the davis dataset"
python test_for_Ablation.py --model HFS --dataset davis
wait
echo "********************************************************"
echo "\n"

echo "##################################################################################################"
echo "END ..."