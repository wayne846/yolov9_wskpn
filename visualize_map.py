import torch
import matplotlib.pyplot as plt
import utils.dataset as wskpn_dataset

def main():
    # 1. 設定裝置與載入權重
    device = torch.device('cuda:0')
    weights_path = 'runs/train/exp2/weights/best.pt'  # 請換成你實際的 best.pt 路徑
    print(f"Loading weights from {weights_path}...")
    
    ckpt = torch.load(weights_path, map_location=device)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float().eval()

    # 2. 準備一張圖片
    database = wskpn_dataset.DataBase()
    val_dataset = wskpn_dataset.BMFRFullResAlDataset(database, use_val=True)
    im, target = val_dataset[0]  # 取得驗證集的第一張圖片
    im = im.unsqueeze(0).to(device)  # 增加 batch 維度變成 (1, 10, 128, 128)

    # 3. 進行推論，取得中間特徵圖
    print("Running inference...")
    with torch.no_grad():
        # 因為我們剛修改了 WSKPNHead，所以它現在會回傳三個變數
        x_out, x_guidemap, x_alpha = model(im)

    # 4. 轉換為 NumPy 以供 matplotlib 畫圖
    # shape: (1, 6, H, W) -> (6, H, W)
    guidemap_np = x_guidemap[0].cpu().numpy()
    alpha_np = x_alpha[0].cpu().numpy()

    # 5. 繪製熱力圖
    print("Plotting heatmaps...")
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    kernel_sizes = [3, 5, 7, 9, 11, 13]  # WSKPN 預設的 6 種尺寸

    for i in range(6):
        # 第一排：繪製 Importance Map (x_guidemap)
        ax1 = axes[0, i]
        im1 = ax1.imshow(guidemap_np[i], cmap='viridis')
        ax1.set_title(f'Importance Map\n(Kernel={kernel_sizes[i]})')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis('off')

        # 第二排：繪製 Alpha 混合權重 (x_alpha)
        ax2 = axes[1, i]
        im2 = ax2.imshow(alpha_np[i], cmap='plasma')
        ax2.set_title(f'Alpha Weight\n(Kernel={kernel_sizes[i]})')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig('wskpn_feature_maps.png', dpi=300)
    print("視覺化結果已成功儲存為 'wskpn_feature_maps.png'！")

if __name__ == '__main__':
    main()