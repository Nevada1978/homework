# 3.2.2 孤立点滤波

## 孤立点滤波原理

由待测物表面的反光特性引起的点云离群点，该类离群点具有少数点孤立的特征。由于扫描过程中对未采集到的点已经进行了标记（NAN），孤立点滤波即检测相邻 NAN 点之间的非 NAN 点的个数 n，如果 n 小于设定阈值（一般为 15，认为 15 个点为一组孤立点），则认为该部分非 NAN 点为孤立点，需要进行滤除。循环遍历每条光刀线上的点，完成上述滤波。

## 孤立点滤波算法

```cpp
//Filter isolate point
int ROWS2=1280;
int COLS2=iPoint.size()/ROWS2;
int ISOLITENUM=IsolateNum;

for (int col=0;col<COLS2;col++)
{
    int row=0;
    PointXYZ point=iPoint[GetIndex(col,row)];
    while(row!=ROWS2)
    {
        if (IsNan2(iPoint[GetIndex(col,row)])==true)
        {
            row++;
        }
        else
        {
            int numContinous=0;
            while(IsNan2(iPoint[GetIndex(col,row)])==false)
            {
                numContinous++;
                row++;
                if (row==ROWS2)
                {
                    break;
                }
            }
            if (numContinous<=ISOLITENUM)
            {
                row=row-numContinous;
                for (int k=0;k<numContinous;k++)
                {
                    iPoint[GetIndex(col,row)].X(0);
                    iPoint[GetIndex(col,row)].Y(0);
                    iPoint[GetIndex(col,row)].Z(0);
                    //SetNan(col,row);
                    row++;
                }
            }
        }
    }
}
```

## 算法说明

1. **参数设置**：
   - `ROWS2=1280`：扫描线数量
   - `COLS2`：每条扫描线的点数
   - `ISOLITENUM`：孤立点阈值，通常设为15

2. **滤波过程**：
   - 逐列遍历点云数据
   - 跳过NAN点，统计连续非NAN点的个数
   - 如果连续点数小于阈值，则将这些点标记为孤立点并清零

3. **滤波效果**：
   - 有效去除由表面反光造成的离群点
   - 保持点云主体结构的完整性
   - 为后续处理提供更干净的数据