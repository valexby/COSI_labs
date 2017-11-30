#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stack>

const static int default_character_width = 25;
const static int default_character_height = 27;
const static int learn_images = 144;
const static std::string directory = "E:\\DSIP\\";
const static int window_size = 150;
int current_character = 0;

#define MIN_WIDTH 4
#define MAX_WIDTH 25
#define MIN_HEIGHT 15
#define MAX_HEIGHT 35

struct Point {
	int y;
	int x;
	Point(int y, int x) : x(x), y(y) {

	}
};
typedef std::vector<Point> Character;
typedef std::vector<Character> Characters;

bool checkCharacterSize(Character character, int imageHeight, int imageWidth) {
	int xMin = imageWidth, xMax = 0, yMin = imageHeight, yMax = 0;
	for (auto it = character.begin(); it != character.end(); it++) {
		if (it->x < xMin) xMin = it->x;
		if (it->x > xMax) xMax = it->x;
		if (it->y < yMin) yMin = it->y;
		if (it->y > yMax) yMax = it->y;
	}
	int width = xMax - xMin;
	int height = yMax - yMin;
	
	return (width >= MIN_WIDTH && width <= MAX_WIDTH && height >= MIN_HEIGHT && height <= MAX_HEIGHT);
}

void erosion(IplImage *image, int radius, int iterations) {
	IplConvKernel *kernel = cvCreateStructuringElementEx(radius * 2 + 1, radius * 2 + 1, radius, radius, CV_SHAPE_ELLIPSE);
	cvErode(image, image, kernel, iterations);
}

void dilation(IplImage *image, int radius, int iterations) {
	IplConvKernel *kernel = cvCreateStructuringElementEx(radius * 2 + 1, radius * 2 + 1, radius, radius, CV_SHAPE_ELLIPSE);
	cvDilate(image, image, kernel, iterations);
}

void increaseContrast(IplImage *image) {
	uchar fmin = 255, fmax = 0;
	for (int i = 0; i < image->height; i++) {
		for (int j = 0; j < image->width; j++) {
			uchar pixel = CV_IMAGE_ELEM(image, uchar, i, j);
			if (pixel < fmin) {
				fmin = pixel;
			}
			if (pixel > fmax) {
				fmax = pixel;
			}
		}
	}

	for (int i = 0; i < image->height; i++) {
		uchar* ptr = (uchar*)(image->imageData + i * image->widthStep);
		for (int j = 0; j < image->width; j++) {
			float t = (float)(ptr[j] - fmin) / (float)(fmax - fmin) * 255;
			ptr[j] = (int)t;
		}
	}
}

int** labelImage(IplImage *image, int &label) {
	int **map = new int*[image->height];
	for (int i = 0; i < image->height; i++) {
		map[i] = new int[image->width];
		for (int j = 0; j < image->width; j++) {
			map[i][j] = 0;
		}
	}

	std::stack<std::vector<int>> stack;
	label = 0;

	for (int i = 0; i < image->height; i++) {
		for (int j = 0; j < image->width; j++) {
			if (CV_IMAGE_ELEM(image, uchar, i, j) == 0) continue;
			if (map[i][j] > 0) continue;

			std::vector<int> vec(2);
			vec[0] = i;
			vec[1] = j;
			stack.push(vec);
			label++;

			while (!stack.empty()) {
				vec = stack.top();
				stack.pop();
				int i1 = vec[0], j1 = vec[1];

				if (i1 != 0 && j1 != 0) {
					if (CV_IMAGE_ELEM(image, uchar, i1 - 1, j1 - 1) == 255 && map[i1 - 1][j1 - 1] == 0) {
						vec[0] = i1 - 1;
						vec[1] = j1 - 1;
						stack.push(vec);
						map[i1 - 1][j1 - 1] = label;
					}
				}

				if (i1 != 0) {
					if (CV_IMAGE_ELEM(image, uchar, i1 - 1, j1) == 255 && map[i1 - 1][j1] == 0) {
						vec[0] = i1 - 1;
						vec[1] = j1;
						stack.push(vec);
						map[i1 - 1][j1] = label;
					}
				}

				if (i1 != 0 && j1 != image->width - 1) {
					if (CV_IMAGE_ELEM(image, uchar, i1 - 1, j1 + 1) == 255 && map[i1 - 1][j1 + 1] == 0) {
						vec[0] = i1 - 1;
						vec[1] = j1 + 1;
						stack.push(vec);
						map[i1 - 1][j1 + 1] = label;
					}
				}

				if (j1 != 0) {
					if (CV_IMAGE_ELEM(image, uchar, i1, j1 - 1) == 255 && map[i1][j1 - 1] == 0) {
						vec[0] = i1;
						vec[1] = j1 - 1;
						stack.push(vec);
						map[i1][j1 - 1] = label;
					}
				}

				if (j1 != image->width - 1) {
					if (CV_IMAGE_ELEM(image, uchar, i1, j1 + 1) == 255 && map[i1][j1 + 1] == 0) {
						vec[0] = i1;
						vec[1] = j1 + 1;
						stack.push(vec);
						map[i1][j1 + 1] = label;
					}
				}

				if (i1 != image->height - 1 && j1 != 0) {
					if (CV_IMAGE_ELEM(image, uchar, i1 + 1, j1 - 1) == 255 && map[i1 + 1][j1 - 1] == 0) {
						vec[0] = i1 + 1;
						vec[1] = j1 - 1;
						stack.push(vec);
						map[i1 + 1][j1 - 1] = label;
					}
				}

				if (i1 != image->height - 1) {
					if (CV_IMAGE_ELEM(image, uchar, i1 + 1, j1) == 255 && map[i1 + 1][j1] == 0) {
						vec[0] = i1 + 1;
						vec[1] = j1;
						stack.push(vec);
						map[i1 + 1][j1] = label;
					}
				}

				if (i1 != image->height - 1 && j1 != image->width - 1) {
					if (CV_IMAGE_ELEM(image, uchar, i1 + 1, j1 + 1) == 255 && map[i1 + 1][j1 + 1] == 0) {
						vec[0] = i1 + 1;
						vec[1] = j1 + 1;
						stack.push(vec);
						map[i1 + 1][j1 + 1] = label;
					}
				}
			}
		}
	}

	return map;
}

Characters getCharacters(IplImage *image) {
	int maxLabel;

	int **map = labelImage(image, maxLabel);
	Characters out(maxLabel);
	for (int i = 0; i < image->height; i++) {
		for (int j = 0; j < image->width; j++) {
			if (map[i][j] != 0) {
				Point p(i, j);
				out[map[i][j] - 1].push_back(p);
			}
		}
	}

	for (int i = 0; i < image->height; i++) {
		delete[] map[i];
	}
	delete[] map;

	return out;
}

Characters getLikelyCharacters(IplImage *image) {
	Characters out = getCharacters(image);
	for (auto it = out.begin(); it != out.end(); ) {
		if (!checkCharacterSize(*it, image->height, image->width)) {
			it = out.erase(it);
		}
		else {
			it++;
		}
	}

	return out;
}

std::vector<int> getBorder(Character character) {
	std::vector<int> out(4);
	out[0] = 192;
	out[1] = 48;
	out[2] = 0;
	out[3] = 0;

	for (auto it = character.begin(); it != character.end(); it++) {
		if (it->x < out[0])
			out[0] = it->x;
		if (it->y < out[1])
			out[1] = it->y;
		if (it->x > out[2])
			out[2] = it->x;
if (it->y > out[3])
out[3] = it->y;
	}

	return out;
}

Characters removeExternalCharacters(IplImage *image) {
	Characters characters = getCharacters(image);
	Character character;
	int max_area = 0;

	for (auto it = characters.begin(); it != characters.end(); it++) {
		if (it->size() > max_area) {
			max_area = (int)it->size();
			character = *it;
		}
	}

	std::vector<int> max_border = getBorder(character);
	std::vector<int> temp_max_border = getBorder(character);
	if (max_border[2] - max_border[0] < (image->width * 3 / 4)) {
		if (max_border[2] > (image->width * 2 / 3)) {
			max_border[0] = max_border[2] - (image->width * 3 / 4);
		}
		else {
			max_border[2] = max_border[0] + (image->width * 3 / 4);
		}
	}
	if (max_border[3] - max_border[1] < (image->height / 2)) {
		if (max_border[3] > (image->height / 2)) {
			max_border[1] = max_border[3] - (image->height / 2);
		}
		else {
			max_border[3] = max_border[1] + (image->height / 2);
		}
	}

	for (auto it = characters.begin(); it != characters.end(); ) {
		std::vector<int> border = getBorder(*it);
		if (border[0] == temp_max_border[0] && border[1] == temp_max_border[1] && border[2] == temp_max_border[2] && border[3] == temp_max_border[3]) {
			it = characters.erase(it);
		}
		else {
			if (border[0] <= max_border[0] || border[1] < max_border[1] || border[2] >= max_border[2] || border[3] > max_border[3]) {
				it = characters.erase(it);
			}
			else {
				it++;
			}
		}
	}

	return characters;
}

void removeConnectingPixels(IplImage *image) {
	bool **map = new bool*[image->height];
	for (int i = 0; i < image->height; i++) {
		map[i] = new bool[image->width];
		for (int j = 0; j < image->width; j++) {
			map[i][j] = false;
		}
	}

	for (int i = 2; i < image->height - 2; i++) {
		for (int j = 2; j < image->width - 2; j++) {
			if (CV_IMAGE_ELEM(image, uchar, i, j) == 255) {
				if (CV_IMAGE_ELEM(image, uchar, i, j - 1) == 0 && CV_IMAGE_ELEM(image, uchar, i, j + 1) == 0) {
					map[i][j] = true;
				}
			}
		}
	}

	for (int i = 2; i < image->height - 2; i++) {
		uchar *ptr = (uchar*)(image->imageData + i * image->widthStep);
		for (int j = 2; j < image->width - 2; j++) {
			if (map[i][j] == true) {
				ptr[j] = 0;
			}
		}
	}
}

void cropImage(IplImage *image, char *path) {
	Characters characters = getCharacters(image);
	Characters order_characters;

	bool not_empty = false;
	std::vector<int> border;

	for (int i = 0; i < image->width; i++) {
		not_empty = false;
		for (int j = 0; j < image->height; j++) {
			if (CV_IMAGE_ELEM(image, uchar, j, i) == 255) {
				not_empty = true;
				break;
			}
		}

		if (not_empty) {
			for (auto it = characters.begin(); it != characters.end(); it++) {
				border = getBorder(*it);
				if (border[0] == i) {
					order_characters.push_back(*it);
					i = border[2] + 1;
					break;
				}
			}
		}
	}

	for (auto it = order_characters.begin(); it != order_characters.end(); it++) {
		border = getBorder(*it);
		cvSetImageROI(image, cvRect(border[0], border[1], border[2] - border[0], border[3] - border[1]));
		IplImage *temp = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
		cvCopy(image, temp, NULL);
		cvResetImageROI(image);

		IplImage *character_image = cvCreateImage(cvSize(default_character_width, default_character_height), temp->depth, temp->nChannels);
		cvResize(temp, character_image, CV_INTER_LINEAR);

		std::string file_name = std::to_string(++current_character) + ".bmp";
		std::string file_path = path + file_name;
		cvSaveImage(file_path.c_str(), character_image);
		cvReleaseImage(&temp);
		cvReleaseImage(&character_image);
	}
}

int main(int argc, char *argv[]) {	
	IplImage *original_image;
	IplImage *without_external_image;
	IplImage *processed_image;
	std::string file_name, file_path;

	if (argc != 3) {
		return 0;
	}

	original_image = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	without_external_image = cvCreateImage(cvGetSize(original_image), 8, 1);

	increaseContrast(original_image);
	cvSmooth(original_image, original_image, CV_GAUSSIAN, 5, 5);
	cvAdaptiveThreshold(original_image, original_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 9, 1);
		
	removeConnectingPixels(original_image);
	Characters characters = removeExternalCharacters(original_image);
	for (auto it = characters.begin(); it != characters.end(); it++) {
		Character ch = *it;
		for (auto it2 = ch.begin(); it2 != ch.end(); it2++) {
			uchar *ptr = (uchar*)(without_external_image->imageData + it2->y * without_external_image->widthStep);
			ptr[it2->x] = 255;
		}
	}
	for (int i = 0; i < without_external_image->height; i++) {
		for (int j = 0; j < without_external_image->width; j++) {
			if (CV_IMAGE_ELEM(without_external_image, uchar, i, j) != 255) {
				uchar *ptr = (uchar*)(without_external_image->imageData + i * without_external_image->widthStep);
				ptr[j] = 0;
			}
		}
	}

	processed_image = cvCreateImage(cvGetSize(without_external_image), 8, 1);

	Characters likely_characters = getLikelyCharacters(without_external_image);
	for (auto it = likely_characters.begin(); it != likely_characters.end(); it++) {
		Character ch = *it;
		for (auto it2 = ch.begin(); it2 != ch.end(); it2++) {
			uchar *ptr = (uchar*)(processed_image->imageData + it2->y * processed_image->widthStep);
			ptr[it2->x] = 255;
		}
	}
	for (int i = 0; i < processed_image->height; i++) {
		for (int j = 0; j < processed_image->width; j++) {
			if (CV_IMAGE_ELEM(processed_image, uchar, i, j) != 255) {
				uchar *ptr = (uchar*)(processed_image->imageData + i * processed_image->widthStep);
				ptr[j] = 0;
			}
		}
	}

	cropImage(processed_image, argv[2]);

	cvReleaseImage(&original_image);
	cvReleaseImage(&without_external_image);
	cvReleaseImage(&processed_image);

	return 0;
}
