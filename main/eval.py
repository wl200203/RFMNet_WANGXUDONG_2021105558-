from misc import *
import os
import numpy as np
from datetime import datetime


pred_dir = "/home/gpuadmin/hds/mirror_rorate/HetNet/Test_model_result/2024-12-06_15-50_best_mae.pth/MSD/MSD_source"
# gt_dir = "/home/gpuadmin/hds/SAM-Adapter-PyTorch/load/PMD/test/masks"  
gt_dir = "/home/gpuadmin/hds/SAM-Adapter-PyTorch/load/MSD/test/masks"  


print(pred_dir)

# def validate_miou(model, val_loader, nums):
#     model.train(False)  # 切换到评估模式，关闭 Dropout 等层
#     total_inter = 0.0
#     total_union = 0.0
#     with torch.no_grad():
#         for image, mask, shape, name in val_loader:
#             image, mask = image.cuda().float(), mask.cuda().float()
#             pred = (pred >= 0.5).float()  # 将预测值二值化，阈值为 0.5

#             # 计算交集和并集
#             intersection = (pred * mask[0]).sum()
#             union = pred.sum() + mask[0].sum() - intersection

#             total_inter += intersection.item()
#             total_union += union.item()
#     # import pdb;pdb.set_trace()
#     model.train(True)  # 恢复到训练模式
#     # 返回平均的 IoU
#     miou= total_inter / (total_union + 1e-6)  # 避免除以零
#     return miou

# for sub_dir in os.listdir(pred_dir):
#     if 0 == len(os.listdir(os.path.join(pred_dir, sub_dir))):
#         continue



iou_l = []
acc_l = []
mae_l = []
ber_l = []
f_measure_p_l = []
f_measure_r_l = []
precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
for name in os.listdir(os.path.join(pred_dir)):
    mask_path = os.path.join(gt_dir, name)
    #print(name)
    gt = get_gt_mask(name, gt_dir)
    # import pdb;pdb.set_trace()
    normalized_pred = get_normalized_predict_mask(name, pred_dir)
    binary_pred = get_binary_predict_mask(name, pred_dir)
    # import pdb; pdb.set_trace()

    if normalized_pred.ndim == 3:
        normalized_pred = normalized_pred[:, :, 0]
    if binary_pred.ndim == 3:
        binary_pred = binary_pred[:, :, 0]
        
    acc_l.append(accuracy_mirror(binary_pred, gt))
    iou_l.append(compute_iou(binary_pred, gt))
    mae_l.append(compute_mae(normalized_pred, gt))
    ber_l.append(compute_ber(binary_pred, gt))

    pred = (255 * normalized_pred).astype(np.uint8)
    gt = (255 * gt).astype(np.uint8)
    p, r = cal_precision_recall(pred, gt)
    for idx, data in enumerate(zip(p, r)):
        p, r = data
        precision_record[idx].update(p)
        recall_record[idx].update(r)
    
    
results = '%s:  mae: %4f, ber: %4f, acc: %4f, iou: %4f, f_measure: %4f' % (
    pred_dir, 
    np.mean(mae_l), 
    np.mean(ber_l), 
    np.mean(acc_l), 
    np.mean(iou_l),
    cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])
)
print(results)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
output_file = f"evaluation_results_{timestamp}.txt" 

with open(output_file, "w") as f:
    f.write("Evaluation Results\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    f.write(results + "\n")
