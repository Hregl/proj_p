#include "markerdetector.h"
#include "clock.h"

SLSMarkerDetector::SLSMarkerDetector()
{

}

SLSMarkerDetector::~SLSMarkerDetector()
{

}

std::string SLSMarkerDetector::detectMarkers(std::vector<std::pair<glm::vec2, float>> &markers, 
	const cv::Mat &image, bool smooth, bool debug)
{
	markers.clear();
	if (image.empty())
		return "标志点图像为空！\n";
	SLSClock clock;
	clock.begin();
	std::string errorInfo;
	try
	{
		cv::Mat grayImage;
		if (image.type() == CV_8UC3)
			cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
		else
			grayImage = image;
		findCircles(markers, grayImage, smooth, debug);
	}
	catch (cv::Exception &exception)
	{
		errorInfo = exception.what();
	}
	clock.end();
	clock.displayInterval("findCircles");
	return errorInfo;
}

void SLSMarkerDetector::findCircles(std::vector<std::pair<glm::vec2, float>> &markers,
	const cv::Mat &image, bool smooth, bool debug)
{
	auto width = image.cols;
	auto height = image.rows;
	cv::Mat mask;
	SLSClock clock;
	clock.begin();
	cv::Mat smoothedImage;
	if (smooth)
		cv::GaussianBlur(image, smoothedImage, cv::Size(3, 3), 0.0);
	else
		smoothedImage = image;
	threshold(smoothedImage, mask, 40, 255, cv::THRESH_BINARY);
	clock.end();
	clock.displayInterval("threshold");
	cv::Mat edge;
	Canny(smoothedImage, edge, 50, 150, 3);
	clock.begin();
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edge, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	clock.end();
	clock.displayInterval("contour");
	clock.begin();
	cv::Mat sobelX;
	cv::Mat sobelY;
	cv::Sobel(image, sobelX, CV_32FC1, 1, 0, 1);
	cv::Sobel(image, sobelY, CV_32FC1, 0, 1, 1);
	clock.end();
	clock.displayInterval("sobel");

	std::vector<cv::RotatedRect> circles;
	std::vector<std::vector<cv::Point>> subpixelContours;
	clock.begin();
	for (const auto &contour : contours)
	{
		if (contour.size() >= 5)
		{
			auto rect = cv::boundingRect(contour);
			if (rect.width > width / 200 || rect.height > width / 200)
			{
				auto area = fabs(cv::contourArea(contour));
				auto length = cv::arcLength(contour, true);
				//应该用拟合误差
				if (area / length > length / SLS::pi / 4.0f * 0.7f)
				{
					auto circle = cv::fitEllipse(contour);
					glm::ivec2 center(circle.center.x + 0.5f, circle.center.y + 0.5f);
					if (center.x >= 0 && center.y >= 0 && center.x < width && center.y < height)
						if (mask.at<uchar>(center.y, center.x) == 255)
						{
							auto inside = true;
							std::vector<cv::Point2f> subpixelPoints;
							for (const auto &point : contour)
							{
								if (point.x < 2 || point.y < 2 ||
									point.x >= width - 2 || point.y >= height - 2)
								{
									inside = false;
									break;
								}
								else
								{
									//auto delta = subpixelFit(point, image,
									//	sobelX.at<float>(point.y, point.x), sobelY.at<float>(point.y, point.x));
									auto delta = subpixelFit(point, image, sobelX, sobelY);
									if (delta.dot(delta) < 1.0f)
										subpixelPoints.push_back(cv::Point2f(point.x, point.y) + delta);
								}
							}
							if (inside && subpixelPoints.size() >= 5)
							{
								subpixelContours.emplace_back();
								for (const auto &subpixelPoint : subpixelPoints)
									subpixelContours.back().push_back(subpixelPoint);
								//printf("(%f %f) ", circle.center.x, circle.center.y);
								circle = cv::fitEllipse(subpixelPoints);
								//printf("(%f %f)\n", circle.center.x, circle.center.y);
								circles.push_back(circle);
							}
						}
				}
			}
		}
	}
	clock.end();
	clock.displayInterval("subpixel");

	std::vector<cv::RotatedRect> cleanCircles;
	std::vector<bool> processed(circles.size(), false);
	for (auto i = 0; i < circles.size(); ++i)
	{
		if (!processed[i])
		{
			auto minArea = circles[i].size.area();
			auto minAreaCircle = circles[i];
			for (auto j = 0; j < circles.size(); ++j)
			{
				if (i != j && !processed[j] && glm::distance(glm::vec2(circles[i].center.x,
					circles[i].center.y), glm::vec2(circles[j].center.x, circles[j].center.y)) < 5.0f)
				{
					if (circles[j].size.area() < minArea)
					{
						minArea = circles[j].size.area();
						minAreaCircle = circles[j];
					}
					processed[j] = true;
				}
			}
			cleanCircles.push_back(minAreaCircle);
			markers.emplace_back(glm::vec2(minAreaCircle.center.x, minAreaCircle.center.y), 
				minArea);
			processed[i] = true;
		}
	}

	if (debug)
	{
		cv::Mat contourImage(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat circleImage(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
		for (auto i = 0; i < subpixelContours.size(); ++i)
			cv::drawContours(contourImage, subpixelContours, i, cv::Scalar(rand() % 255, 
				rand() % 255, rand() % 255), 1);
		for (const auto &circle : cleanCircles)
			cv::ellipse(circleImage, circle, cv::Scalar(255));
		imwrite("mask.png", mask);
		imwrite("contour.png", contourImage);
		imwrite("circle.png", circleImage);
	}
}

cv::Point2f SLSMarkerDetector::subpixelFit(const cv::Point &point, const cv::Mat &image,
	const cv::Mat &sobelX, const cv::Mat &sobelY)
{
	auto sumWeightsX = 0.0;
	auto sumWeightsY = 0.0;
	auto sumX = 0.0;
	auto sumY = 0.0;
	auto differenceX = sobelX.at<float>(point.y, point.x);
	auto differenceY = sobelY.at<float>(point.y, point.x);
	auto computeSum = [&sumX, &sumY, &sumWeightsX, &sumWeightsY, &sobelX, &sobelY, &point]
		(const int &x, const int &y)
	{
		sumX += sobelX.at<float>(y, x) * (x - point.x);
		sumY += sobelY.at<float>(y, x) * (y - point.y);
		sumWeightsX += sobelX.at<float>(y, x);
		sumWeightsY += sobelY.at<float>(y, x);
	};
	auto angle = glm::degrees(atan2f(differenceY, differenceX));
	if ((angle > 45.0f && angle < 135.0f) || (angle > -135.0f && angle < -45.0f))
		for (auto y = point.y - 2; y <= point.y + 2; ++y)
			computeSum(point.x, y);
	else
		for (auto x = point.x - 2; x <= point.x + 2; ++x)
			computeSum(x, point.y);
	return cv::Point2f(sumX / sumWeightsX, sumY / sumWeightsY);
}

cv::Point2f SLSMarkerDetector::subpixelFit(const cv::Point &point, const cv::Mat &image,
	float differentX, float differentY)
{
	double a[5][5];//存储周边25个点的灰度值；
	cv::Mat maty(25, 1, CV_64F);

	for (int i = 2; i >= -2; i--)
		for (int j = -2; j <= 2; j++)
			a[2 - i][j + 2] = image.at<uchar>(point.y + i, point.x + j);

	int t = 0;
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
		{
			maty.at<double>(t, 0) = a[i][j];
			t++;
		}
	//std::cout << maty << std::endl;
	cv::Mat matb(25, 10, CV_64F);
	double tempx;
	double tempy;
	int i = 0;
	for (int k = 2; k >= -2; k--) {
		for (int j = -2; j <= 2; j++) {
			tempx = j;
			tempy = k;

			matb.at<double>(i, 0) = 1;
			matb.at<double>(i, 1) = tempx;
			matb.at<double>(i, 2) = tempy;
			matb.at<double>(i, 3) = tempx * tempx;
			matb.at<double>(i, 4) = tempx * tempy;
			matb.at<double>(i, 5) = tempy * tempy;
			matb.at<double>(i, 6) = tempx * tempx * tempx;
			matb.at<double>(i, 7) = tempx * tempx * tempy;
			matb.at<double>(i, 8) = tempy * tempy * tempx;
			matb.at<double>(i, 9) = tempy * tempy * tempy;
			i++;

		}
	}

	cv::Mat matc;
	solve(matb, maty, matc, cv::DECOMP_SVD);
                    
	 //计算梯度
	double k[10];
	for (int i = 0; i < 10; i++)
		k[i] = matc.at<double>(i, 0);

	double gradX = differentX;// k[1];
	double gradY = differentY;// k[2];
	double angle = atanf(gradY / gradX);
	//  std::cout << cc << std::endl;
	double p = ((k[3] * cos(angle) * cos(angle) + k[4] * cos(angle) * sin(angle) + k[5] * sin(angle) * sin(angle))
		/ (k[6] * cos(angle) * cos(angle) * cos(angle) + k[7] * cos(angle) * cos(angle) * sin(angle) + k[8] * cos(angle) * sin(angle) * sin(angle) + k[9] * sin(angle) * sin(angle) * sin(angle))) * (-1.0 / 3.0);
	// std::cout << "sin" << sin(angle) << std::endl;

	return cv::Point2f(p * cos(angle), p * sin(angle));
}