#include "matrix.h"

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
   data.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(int rows, int cols, const std::vector<std::vector<double>> &data) : rows(rows), cols(cols), data(data) {
   if (data.size() != rows || (rows > 0 && data[0].size() != cols)) {
      throw std::invalid_argument("Incorrect data dimensions!");
   }
}

Matrix::Matrix(emscripten::val v) {
   unsigned int length = v["length"].as<unsigned int>();
   if (length == 0) {
      rows = 0;
      cols = 0;
      return;
   }

   rows = length;
   emscripten::val firstRow = v[0];
   cols = firstRow["length"].as<unsigned int>();

   data.resize(rows);
   for (unsigned int i = 0; i < rows; ++i) {
      data[i].resize(cols);
      emscripten::val row = v[i];
      if (row["length"].as<unsigned int>() != cols) {
         throw std::invalid_argument("Inconsistent row lengths in JS array");
      }
      for (unsigned int j = 0; j < cols; ++j) {
         data[i][j] = row[j].as<double>();
      }
   }
}

int Matrix::getRows() const {
   return rows;
}

int Matrix::getCols() const {
   return cols;
}

std::vector<std::vector<double>> Matrix::getData() const {
   return data;
}

void Matrix::setData(const std::vector<std::vector<double>> &newData) {
   data = newData;
   rows = data.size();
   cols = (rows > 0) ? data[0].size() : 0;
}

double &Matrix::at(int i, int j) {
   return data[i][j];
}

const double &Matrix::at(int i, int j) const {
   return data[i][j];
}

void Matrix::randomWeights() {
   // Use static generator to avoid re-initialization cost
   static std::random_device rd;
   static std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(-1.0, 1.0);

   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         data[i][j] = dis(gen);
      }
   }
}

void Matrix::randomWeights(bool round) {
   static std::random_device rd;
   static std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(-1.0, 1.0);

   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         double val = dis(gen);
         if (round) {
            // Round to 1 decimal place like in JS: parseFloat((Math.random() * 2 - 1).toFixed(1))
            val = std::round(val * 10.0) / 10.0;
         }
         data[i][j] = val;
      }
   }
}

Matrix Matrix::add(const Matrix &m1, const Matrix &m2) {
   checkDimensions(m1, m2);
   Matrix temp(m1.rows, m1.cols);
   for (int i = 0; i < m1.rows; i++) {
      for (int j = 0; j < m1.cols; j++) {
         temp.data[i][j] = m1.data[i][j] + m2.data[i][j];
      }
   }
   return temp;
}

void Matrix::add(const Matrix &m2) {
   checkDimensions(*this, m2);
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         data[i][j] += m2.data[i][j];
      }
   }
}

void Matrix::add(double scalar) {
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         data[i][j] += scalar;
      }
   }
}

Matrix Matrix::subtract(const Matrix &m1, const Matrix &m2) {
   checkDimensions(m1, m2);
   Matrix temp(m1.rows, m1.cols);
   for (int i = 0; i < m1.rows; i++) {
      for (int j = 0; j < m1.cols; j++) {
         temp.data[i][j] = m1.data[i][j] - m2.data[i][j];
      }
   }
   return temp;
}

void Matrix::subtract(const Matrix &m2) {
   checkDimensions(*this, m2);
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         data[i][j] -= m2.data[i][j];
      }
   }
}

Matrix Matrix::multiply(const Matrix &m1, const Matrix &m2) {
   checkDimensions(m1, m2);
   Matrix temp(m1.rows, m1.cols);
   for (int i = 0; i < m1.rows; i++) {
      for (int j = 0; j < m1.cols; j++) {
         temp.data[i][j] = m1.data[i][j] * m2.data[i][j];
      }
   }
   return temp;
}

void Matrix::multiply(const Matrix &m2) {
   checkDimensions(*this, m2);
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         data[i][j] *= m2.data[i][j];
      }
   }
}

void Matrix::multiply(double scalar) {
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         data[i][j] *= scalar;
      }
   }
}

Matrix Matrix::dot(const Matrix &m1, const Matrix &m2) {
   if (m1.cols != m2.rows) {
      throw std::invalid_argument("Matrixes are not dot compatible!");
   }
   Matrix temp(m1.rows, m2.cols);
   for (int i = 0; i < temp.rows; i++) {
      for (int j = 0; j < temp.cols; j++) {
         double sum = 0;
         for (int k = 0; k < m1.cols; k++) {
            sum += m1.data[i][k] * m2.data[k][j];
         }
         temp.data[i][j] = sum;
      }
   }
   return temp;
}

Matrix Matrix::convertFromArray(const std::vector<double> &arr) {
   std::vector<std::vector<double>> d;
   d.push_back(arr);
   return Matrix(1, arr.size(), d);
}

Matrix Matrix::map(const Matrix &m1, std::function<double(double)> func) {
   Matrix temp(m1.rows, m1.cols);
   for (int i = 0; i < m1.rows; i++) {
      for (int j = 0; j < m1.cols; j++) {
         temp.data[i][j] = func(m1.data[i][j]);
      }
   }
   return temp;
}

void Matrix::map(std::function<double(double)> func) {
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         data[i][j] = func(data[i][j]);
      }
   }
}

Matrix Matrix::transpose(const Matrix &m1) {
   Matrix temp(m1.cols, m1.rows);
   for (int i = 0; i < m1.rows; i++) {
      for (int j = 0; j < m1.cols; j++) {
         temp.data[j][i] = m1.data[i][j];
      }
   }
   return temp;
}

void Matrix::transpose() {
   Matrix temp(cols, rows);
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         temp.data[j][i] = data[i][j];
      }
   }
   *this = temp;
}

void Matrix::checkDimensions(const Matrix &m1, const Matrix &m2) {
   if (m1.rows != m2.rows || m1.cols != m2.cols) {
      throw std::invalid_argument("Matrixes are of different dimensions!");
   }
}

void Matrix::show() const {
   std::cout << "\033[1;34m-------------------\033[0m" << std::endl;
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         if (j == cols - 1)
            std::cout << data[i][j];
         else
            std::cout << data[i][j] << " \033[1;34m|\033[0m ";
      }
      std::cout << std::endl;
   }
}
