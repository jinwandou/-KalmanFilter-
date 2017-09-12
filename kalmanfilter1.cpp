#include "opencv2/video/tracking.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <iostream>  
#include <stdio.h>  
using namespace std;  
using namespace cv;  
  
//计算相对窗口的坐标值，因为坐标原点在左上角，所以sin前有个负号  
static inline Point calcPoint(Point2f center, double R, double angle)  
{  
    return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;  
}  
  
static void help()  
{  
    printf( "\nExamle of c calls to OpenCV's Kalman filter.\n"  
"   Tracking of rotating point.\n"  
"   Rotation speed is constant.\n"  
"   Both state and measurements vectors are 1D (a point angle),\n"  
"   Measurement is the real point angle + gaussian noise.\n"  
"   The real and the estimated points are connected with yellow line segment,\n"  
"   the real and the measured points are connected with red line segment.\n"  
"   (if Kalman filter works correctly,\n"  
"    the yellow segment should be shorter than the red one).\n"  
            "\n"  
"   Pressing any key (except ESC) will reset the tracking with a different speed.\n"  
"   Pressing ESC will stop the program.\n"  
            );  
}  
  
int main(int, char**)  
{  
    help();  
    Mat img(500, 500, CV_8UC3);  
    KalmanFilter KF(2, 1, 0);                                    //创建卡尔曼滤波器对象KF  
    Mat state(2, 1, CV_32F);                                     //state(角度，△角度)  
    Mat processNoise(2, 1, CV_32F);  
    Mat measurement = Mat::zeros(1, 1, CV_32F);                 //定义测量值  
    char code = (char)-1;  
  
    for(;;)  
    {  
        //1.初始化  
        randn( state, Scalar::all(0), Scalar::all(0.1) );          //  
        KF.transitionMatrix = *(Mat_<float>(2, 2) << 1, 1, 0, 1);  //转移矩阵A[1,1;0,1]      
          
  
        //将下面几个矩阵设置为对角阵  
        setIdentity(KF.measurementMatrix);                             //测量矩阵H  
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));            //系统噪声方差矩阵Q  
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));        //测量噪声方差矩阵R  
        setIdentity(KF.errorCovPost, Scalar::all(1));                  //后验错误估计协方差矩阵P  
  
        randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));          //x(0)初始化  
          
        for(;;)  
        {  
            Point2f center(img.cols*0.5f, img.rows*0.5f);          //center图像中心点  
            float R = img.cols/3.f;                                //半径  
            double stateAngle = state.at<float>(0);                //跟踪点角度  
            Point statePt = calcPoint(center, R, stateAngle);     //跟踪点坐标statePt  
  
            //2. 预测  
            Mat prediction = KF.predict();                       //计算预测值，返回x'  
            double predictAngle = prediction.at<float>(0);          //预测点的角度  
            Point predictPt = calcPoint(center, R, predictAngle);   //预测点坐标predictPt  
  
  
            //3.更新  
            //measurement是测量值  
            randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));     //给measurement赋值N(0,R)的随机值  
  
            // generate measurement  
            measurement += KF.measurementMatrix*state;  //z = z + H*x;  
              
            double measAngle = measurement.at<float>(0);  
            Point measPt = calcPoint(center, R, measAngle);  
  
            // plot points  
            //定义了画十字的方法，值得学习下  
            #define drawCross( center, color, d )                                 \  
                line( img, Point( center.x - d, center.y - d ),                \  
                             Point( center.x + d, center.y + d ), color, 1, CV_AA, 0); \  
                line( img, Point( center.x + d, center.y - d ),                \  
                             Point( center.x - d, center.y + d ), color, 1, CV_AA, 0 )  
  
            img = Scalar::all(0);  
            drawCross( statePt, Scalar(255,255,255), 3 );  
            drawCross( measPt, Scalar(0,0,255), 3 );  
            drawCross( predictPt, Scalar(0,255,0), 3 );  
            line( img, statePt, measPt, Scalar(0,0,255), 3, CV_AA, 0 );  
            line( img, statePt, predictPt, Scalar(0,255,255), 3, CV_AA, 0 );  
  
  
            //调用kalman这个类的correct方法得到加入观察值校正后的状态变量值矩阵  
            if(theRNG().uniform(0,4) != 0)  
                KF.correct(measurement);  
  
            //不加噪声的话就是匀速圆周运动，加了点噪声类似匀速圆周运动，因为噪声的原因，运动方向可能会改变  
            randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));   //vk  
            state = KF.transitionMatrix*state + processNoise;     
  
            imshow( "Kalman", img );  
            code = (char)waitKey(100);  
  
            if( code > 0 )  
                break;  
        }  
        if( code == 27 || code == 'q' || code == 'Q' )  
            break;  
    }  
  
    return 0;  
}  
