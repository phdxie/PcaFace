#ifndef __PCA_H__
#define __PCA_H__


#include "opencv.hpp"

#define LOG_FILE "..\\result\\log.txt"



class PCAAlg
{
public:
	Mat EuclidDistClassifier(Mat TestData);
    Mat NormalizeTestData(Mat TestData);

	void InitTrainData( Mat TrainData);
	void CalDataMean(Mat TrainData);
	void CalEigenVector();
	void Trainer();
	Mat GetDataMean() { return m_MeanData; }
	Mat GetRawEigenVector() { return m_RawEigenVector; }
	Mat GetEigenVector() { return m_NormalEigenVector; }
	Mat GetTrainDataDimReduced() ;
	int GetTrainNum() { return m_iTrainNum; }
	Mat ReconstructData(Mat Data);
private:
	int m_iTrainNum;
	Mat m_TrainData;
	Mat m_MeanData;
	Mat m_NormalEigenVector;
	Mat m_RawEigenVector;
	Mat m_DataDimReduced;
	Mat m_fMax;
	Mat m_fMin;
};

#endif

