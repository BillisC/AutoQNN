#ifndef TEST_SUITES_H
#define TEST_SUITES_H

void test_buffer();
void test_camera();
void test_tensor();
void test_casts();
void test_model(const char *model_path, const char *missing_symbols_path);

#endif
