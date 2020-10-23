#include <iostream>
#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include"opencv2/imgproc/imgproc.hpp"
#include <stdio.h>

using namespace std;
using namespace cv;

//拓展为dft最佳图片尺寸
Mat expandDftPic(Mat img) {
    //获得DFT最佳长宽
    int dftRows = getOptimalDFTSize(img.rows);
    int dftCols = getOptimalDFTSize(img.cols);
    //拓展填充
    Mat padded;
    copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));
    return padded;
}
//图片中心化
Mat centerlize(Mat complex,int cx,int cy) {
    Mat planes[] = { complex ,complex };
    split(complex, planes);
    Mat q0Re(planes[0], Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
    Mat q1Re(planes[0], Rect(cx, 0, cx, cy));      //右上角图像
    Mat q2Re(planes[0], Rect(0, cy, cx, cy));      //左下角图像
    Mat q3Re(planes[0], Rect(cx, cy, cx, cy));     //右下角图像
    //交换左上右下象限
    Mat tmp;
    q0Re.copyTo(tmp);
    q3Re.copyTo(q0Re);
    tmp.copyTo(q3Re);
    //变换右上左下象限
    q1Re.copyTo(tmp);
    q2Re.copyTo(q1Re);
    tmp.copyTo(q2Re);
    //Im
    Mat q0Im(planes[1], Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
    Mat q1Im(planes[1], Rect(cx, 0, cx, cy));      //右上角图像
    Mat q2Im(planes[1], Rect(0, cy, cx, cy));      //左下角图像
    Mat q3Im(planes[1], Rect(cx, cy, cx, cy));     //右下角图像
    //交换左上右下象限
    q0Im.copyTo(tmp);
    q3Im.copyTo(q0Im);
    tmp.copyTo(q3Im);
    //变换右上左下象限
    q1Im.copyTo(tmp);
    q2Im.copyTo(q1Im);
    tmp.copyTo(q2Im);
    //合并通道
    Mat centerComplex;
    merge(planes, 2, centerComplex);
    return centerComplex;

}
//傅里叶变换，得到复数域图像
Mat myDft(Mat padded) {
    //创建虚部为0的图片
    Mat dftPlanes[] = { Mat_<float>(padded),Mat::zeros(padded.size(), CV_32F) };
    //合并实部虚部
    Mat complexI;
    merge(dftPlanes, 2, complexI);
    //dft
    dft(complexI, complexI);
    return complexI;
}
//将复数域图像反变换得到原图，由于是幅值计算，因此不用再将图像的左上右下，右上左下交换
Mat myIdft(Mat complexI) {
    Mat idftcvt;
    //反变换
    idft(complexI, idftcvt);
    //分离通道
    Mat planes[] = { idftcvt ,idftcvt };
    split(idftcvt, planes);
    //计算幅值
    Mat dst;
    magnitude(planes[0], planes[1], dst);
    //归一化
    normalize(dst, dst, 1, 0, NORM_MINMAX);
    return dst;
}
//从复数域图像计算幅值得到频域图像
Mat showDFT(Mat complexI) {
    //创建实平面和虚平面{实平面，虚平面（全0）}
    Mat planes[] = { Mat_<float>(complexI), Mat::zeros(complexI.size(), CV_32F) };
    //将complex分离,planxes[0]为实部，planes[1]为虚部
    split(complexI, planes);
    //计算实部与虚部的赋值,sqrt(Re*Re+Im*Im)
    Mat magI;
    magnitude(planes[0], planes[1], magI);
    //转换到对数尺度
    ///全加1，用于对数变换，M = log(1+M)
    magI += Scalar::all(1);
    ///对数变换
    log(magI, magI);
    //归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    return magI;
}

//生成理想低通滤波器模板
Mat  ideal_lbrf_kernel(Mat img, float D0) {
    Mat ideal_low_pass(img.size(), CV_32FC1);
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            float d = sqrt(pow(x - img.rows / 2, 2) + pow(y - img.cols / 2, 2));
            if (d <= D0) {
                ideal_low_pass.at<float>(x, y) = 1;
            }
            else {
                ideal_low_pass.at<float>(x, y) = 0;
            }
        }
    }
    imshow("理想低通滤波器", ideal_low_pass);
    return ideal_low_pass;
}
//低通滤波函数
Mat myLowFliter(Mat& img, float D0) {
    if (img.empty()) {
        cerr << "input picture NULL" << endl;
    }
    imshow("原图", img);
    //拓展图片
    Mat padded = expandDftPic(img);
    //dft
    Mat complexI = myDft(padded);
    //原图频谱图
    Mat imgShow = showDFT(complexI);
    imshow("原图频谱图", imgShow);
    //傅里叶逆变换
    Mat complexI_cvt = myIdft(complexI);
    imshow("原图傅里叶逆变换", complexI_cvt);
    //分离通道
    Mat dftPlanes[] = { complexI ,complexI };
    split(complexI, dftPlanes);
    //裁剪，中心化
    dftPlanes[0] = dftPlanes[0](Rect(0, 0, dftPlanes[0].cols & -2, dftPlanes[0].rows & -2));
    int cx = dftPlanes[0].cols / 2;
    int cy = dftPlanes[0].rows / 2;
    Mat centerComplex = centerlize(complexI, cx, cy);
    Mat imgCenterShow = showDFT(centerComplex);
    imshow("中心化频谱图", imgCenterShow);

    //产生低通滤波器
    Mat idealLowPass = ideal_lbrf_kernel(padded, D0);
    //将中心图像分离通道
    Mat planes[] = { centerComplex ,centerComplex };
    split(centerComplex, planes);
    //低通滤波
    Mat blurRe, blurIm, blurDft;
    multiply(planes[0], idealLowPass, blurRe);
    multiply(planes[1], idealLowPass, blurIm);
    //合并通道
    Mat blurPlanes[] = { blurRe, blurIm };
    merge(blurPlanes, 2, blurDft);
    //显示滤波后频谱图
    Mat blurShow = showDFT(blurDft);
    imshow("低通滤波频谱图", blurShow);
    //IDFT
    Mat dst = myIdft(blurDft);
    imshow("结果图", dst);
    return dst;
}

int main() {
    Mat img = imread("C://Users//Chrysanthemum//Desktop//1.png", 0);
    float D0 = 100;

    Mat dst = myLowFliter(img, D0);

    waitKey(0);

    return 0;
}