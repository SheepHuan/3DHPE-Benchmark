import matplotlib.pyplot as plt

tabs = {
    "D3HP (1 view)": {
        "x": [43.42354389],
        "y": [19.6],
        "marker": '.',
    },
    "Mobi3DHPE (1 view)": {
        "x": [0.505615091],
        "y": [51.4],
        "marker": 's',
    },
    "SRNet (1 view)":{
        "x": [28.92569668],
        "y": [32], 
        "marker": 'o',
    },
    "JointFormer (1 view)":{
        "x": [28.97733397],
        "y": [34], 
        "marker": '+',
    },
    "PoseNet3d (1 view)" :{
        "x": [12.37523494],
        "y": [53.3], 
        "marker": '*',       
    },
    "LTHP (4 views)":{
        "x": [236.6133793],
        "y": [17.7], 
        "marker": 'D',
    },
    "CrossFusion (4 views)":{
        "x": [87.28615322],
        "y": [26.21], 
        "marker": '^',
    }
}

for label,item in tabs.items():
    plt.scatter(item["x"],item["y"],marker=item["marker"],label=label)

# 添加标题和坐标轴标签
# plt.title('Scatter Plot')
plt.xlabel('Macs (G)')
plt.ylabel('MPJPE (ms)')
plt.legend()
# 显示图形

# 保存图片，并设置DPI为300
# plt.savefig('scatter_plot.png',)
plt.savefig("1.png", dpi=500)
