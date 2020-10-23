#include <iostream>
#include<opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include"opencv2/imgproc/imgproc.hpp"
#include <stdio.h>

using namespace std;
using namespace cv;

//��չΪdft���ͼƬ�ߴ�
Mat expandDftPic(Mat img) {
    //���DFT��ѳ���
    int dftRows = getOptimalDFTSize(img.rows);
    int dftCols = getOptimalDFTSize(img.cols);
    //��չ���
    Mat padded;
    copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));
    return padded;
}
//ͼƬ���Ļ�
Mat centerlize(Mat complex,int cx,int cy) {
    Mat planes[] = { complex ,complex };
    split(complex, planes);
    Mat q0Re(planes[0], Rect(0, 0, cx, cy));       //���Ͻ�ͼ�񻮶�ROI����
    Mat q1Re(planes[0], Rect(cx, 0, cx, cy));      //���Ͻ�ͼ��
    Mat q2Re(planes[0], Rect(0, cy, cx, cy));      //���½�ͼ��
    Mat q3Re(planes[0], Rect(cx, cy, cx, cy));     //���½�ͼ��
    //����������������
    Mat tmp;
    q0Re.copyTo(tmp);
    q3Re.copyTo(q0Re);
    tmp.copyTo(q3Re);
    //�任������������
    q1Re.copyTo(tmp);
    q2Re.copyTo(q1Re);
    tmp.copyTo(q2Re);
    //Im
    Mat q0Im(planes[1], Rect(0, 0, cx, cy));       //���Ͻ�ͼ�񻮶�ROI����
    Mat q1Im(planes[1], Rect(cx, 0, cx, cy));      //���Ͻ�ͼ��
    Mat q2Im(planes[1], Rect(0, cy, cx, cy));      //���½�ͼ��
    Mat q3Im(planes[1], Rect(cx, cy, cx, cy));     //���½�ͼ��
    //����������������
    q0Im.copyTo(tmp);
    q3Im.copyTo(q0Im);
    tmp.copyTo(q3Im);
    //�任������������
    q1Im.copyTo(tmp);
    q2Im.copyTo(q1Im);
    tmp.copyTo(q2Im);
    //�ϲ�ͨ��
    Mat centerComplex;
    merge(planes, 2, centerComplex);
    return centerComplex;

}
//����Ҷ�任���õ�������ͼ��
Mat myDft(Mat padded) {
    //�����鲿Ϊ0��ͼƬ
    Mat dftPlanes[] = { Mat_<float>(padded),Mat::zeros(padded.size(), CV_32F) };
    //�ϲ�ʵ���鲿
    Mat complexI;
    merge(dftPlanes, 2, complexI);
    //dft
    dft(complexI, complexI);
    return complexI;
}
//��������ͼ�񷴱任�õ�ԭͼ�������Ƿ�ֵ���㣬��˲����ٽ�ͼ����������£��������½���
Mat myIdft(Mat complexI) {
    Mat idftcvt;
    //���任
    idft(complexI, idftcvt);
    //����ͨ��
    Mat planes[] = { idftcvt ,idftcvt };
    split(idftcvt, planes);
    //�����ֵ
    Mat dst;
    magnitude(planes[0], planes[1], dst);
    //��һ��
    normalize(dst, dst, 1, 0, NORM_MINMAX);
    return dst;
}
//�Ӹ�����ͼ������ֵ�õ�Ƶ��ͼ��
Mat showDFT(Mat complexI) {
    //����ʵƽ�����ƽ��{ʵƽ�棬��ƽ�棨ȫ0��}
    Mat planes[] = { Mat_<float>(complexI), Mat::zeros(complexI.size(), CV_32F) };
    //��complex����,planxes[0]Ϊʵ����planes[1]Ϊ�鲿
    split(complexI, planes);
    //����ʵ�����鲿�ĸ�ֵ,sqrt(Re*Re+Im*Im)
    Mat magI;
    magnitude(planes[0], planes[1], magI);
    //ת���������߶�
    ///ȫ��1�����ڶ����任��M = log(1+M)
    magI += Scalar::all(1);
    ///�����任
    log(magI, magI);
    //��һ��������0-1֮��ĸ�����������任Ϊ���ӵ�ͼ���ʽ
    normalize(magI, magI, 0, 1, NORM_MINMAX);

    return magI;
}

//���������ͨ�˲���ģ��
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
    imshow("�����ͨ�˲���", ideal_low_pass);
    return ideal_low_pass;
}
//��ͨ�˲�����
Mat myLowFliter(Mat& img, float D0) {
    if (img.empty()) {
        cerr << "input picture NULL" << endl;
    }
    imshow("ԭͼ", img);
    //��չͼƬ
    Mat padded = expandDftPic(img);
    //dft
    Mat complexI = myDft(padded);
    //ԭͼƵ��ͼ
    Mat imgShow = showDFT(complexI);
    imshow("ԭͼƵ��ͼ", imgShow);
    //����Ҷ��任
    Mat complexI_cvt = myIdft(complexI);
    imshow("ԭͼ����Ҷ��任", complexI_cvt);
    //����ͨ��
    Mat dftPlanes[] = { complexI ,complexI };
    split(complexI, dftPlanes);
    //�ü������Ļ�
    dftPlanes[0] = dftPlanes[0](Rect(0, 0, dftPlanes[0].cols & -2, dftPlanes[0].rows & -2));
    int cx = dftPlanes[0].cols / 2;
    int cy = dftPlanes[0].rows / 2;
    Mat centerComplex = centerlize(complexI, cx, cy);
    Mat imgCenterShow = showDFT(centerComplex);
    imshow("���Ļ�Ƶ��ͼ", imgCenterShow);

    //������ͨ�˲���
    Mat idealLowPass = ideal_lbrf_kernel(padded, D0);
    //������ͼ�����ͨ��
    Mat planes[] = { centerComplex ,centerComplex };
    split(centerComplex, planes);
    //��ͨ�˲�
    Mat blurRe, blurIm, blurDft;
    multiply(planes[0], idealLowPass, blurRe);
    multiply(planes[1], idealLowPass, blurIm);
    //�ϲ�ͨ��
    Mat blurPlanes[] = { blurRe, blurIm };
    merge(blurPlanes, 2, blurDft);
    //��ʾ�˲���Ƶ��ͼ
    Mat blurShow = showDFT(blurDft);
    imshow("��ͨ�˲�Ƶ��ͼ", blurShow);
    //IDFT
    Mat dst = myIdft(blurDft);
    imshow("���ͼ", dst);
    return dst;
}

int main() {
    Mat img = imread("C://Users//Chrysanthemum//Desktop//1.png", 0);
    float D0 = 100;

    Mat dst = myLowFliter(img, D0);

    waitKey(0);

    return 0;
}