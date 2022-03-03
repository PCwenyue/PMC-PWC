"Parallel multiscale context-based edge-preserving optical flow estimation with occlusion detection"

author:CongxuanZhang, ChengFeng, ZhenChen, WeimingHu, MingLi
Key Laboratory of Nondestructive Testing, Ministry of Education, Nanchang Hangkong University, Nanchang 330063, China
Institute of Automation, Chinese Academy of Sciences, Beijing 100190, China

paper link:https://www.sciencedirect.com/science/article/pii/S0923596521002770?via%3Dihub

Evaluation (PMC_PWC in MPI-Sintel Training Full)
sh scripts/validation.sh i_pmca_p ./weights/PMC_PWC_SINTEL.pt SintelTrainingCombFull /path/MPI-Sintel-complete MultiScaleEPE_PWC_Bi_Occ_upsample True

Evaluation (PMC_PWC in MPI-Sintel Training Clean)
sh scripts/validation.sh i_pmca_p ./weights/PMC_PWC_SINTEL.pt SintelTrainingCleanFull /path/MPI-Sintel-complete MultiScaleEPE_PWC_Bi_Occ_upsample True

Evaluation (PMC_PWC in MPI-Sintel Training Final)
sh scripts/validation.sh i_pmca_p ./weights/PMC_PWC_SINTEL.pt SintelTrainingCleanFull /path/MPI-Sintel-complete MultiScaleEPE_PWC_Bi_Occ_upsample True

Evaluation (PMC_PWC in KITTI2015 Training)
sh scripts/validation.sh i_pmca_p ./weights/PMC_PWC_KITTI.pt KittiComb2015Full /home/fengcheng970pro/Documents/dataset/KITTI MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI True

Test (PMC_PWC in MPI-Sintel Training Clean)
sh scripts/test.sh i_pmca_p ./weights/PMC_PWC_SINTEL.pt SintelTrainingCleanFull /home/fengcheng970pro/Documents/dataset/MPI-Sintel-complete ./test True True

Test (PMC_PWC in MPI-Sintel Training Final)
sh scripts/test.sh i_pmca_p ./weights/PMC_PWC_SINTEL.pt SintelTrainingFinalFull /home/fengcheng970pro/Documents/dataset/MPI-Sintel-complete ./test True True

Test (PMC_PWC in MPI-Sintel Test Clean)
sh scripts/test.sh i_pmca_p ./weights/PMC_PWC_SINTEL.pt SintelTestClean /home/fengcheng970pro/Documents/dataset/MPI-Sintel-complete ./test True True

Test (PMC_PWC in MPI-Sintel Test Final)
sh scripts/test.sh i_pmca_p ./weights/PMC_PWC_SINTEL.pt SintelTestFinal /home/fengcheng970pro/Documents/dataset/MPI-Sintel-complete ./test True True

Test (PMC_PWC in KITTI Training)
sh scripts/test.sh i_pmca_p ./weights/PMC_PWC_KITTI.pt KittiComb2015Full /home/fengcheng970pro/Documents/dataset/KITTI ./test_kitti2015Training True True

Test (PMC_PWC in KITTI Test)
sh scripts/test.sh i_pmca_p ./weights/PMC_PWC_KITTI.pt KittiComb2015Test /home/fengcheng970pro/Documents/dataset/KITTI ./test_kitti2015Test True True
