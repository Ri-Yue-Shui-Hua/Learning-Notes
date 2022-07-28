#include "YKmeans.h"
#include <vector>
#include <iostream>
#include <math.h>
using namespace std;

const int pointsCount = 9;
const int clusterCount = 2;

int main()
{
	//构造待聚类数据集
	vector< vector<double> > data;
	vector<double> points[pointsCount];
	for (int i = 0; i < pointsCount; i++)
	{
		points[i].push_back(i);
		points[i].push_back(i*i);
		points[i].push_back(sqrt(i));
		data.push_back(points[i]);
	}
	
	//构建聚类算法
	KMEANS<double> kmeans;
	//数据加载入算法
	kmeans.loadData(data);
	//运行k均值聚类算法
	kmeans.kmeans(clusterCount);
	
	//输出聚类后各点所属类情况
	for (int i = 0; i < pointsCount; i++)
		cout << kmeans.clusterAssment[i].minIndex << endl;
	//输出类中心
	cout << endl << endl;
	for (int i = 0; i < clusterCount; i++)
	{
		for (int j = 0; j < 3; j++)
			cout << kmeans.centroids[i][j] << ',' << '\t' ;
		cout << endl;
	}
	
	return(0);
}

