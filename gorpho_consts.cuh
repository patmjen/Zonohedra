#ifndef __GORPHO_CONSTS_H__
#define __GORPHO_CONSTS_H__

// TODO: More nuanced result values
enum GorphoResult {
    GORPHO_SUCCESS = 0,
    GORPHO_FAILURE
};

enum GorphoOp {
    GORPHO_DILATE,
    GORPHO_ERODE
};

enum GorphoDir {
    GORPHO_DIR_X     = 0x0010,
    GORPHO_DIR_X_NEG = 0x0011,
    GORPHO_DIR_X_POS = 0x0012,
    GORPHO_DIR_Y     = 0x0100,
    GORPHO_DIR_Y_POS = 0x0101,
    GORPHO_DIR_Y_NEG = 0x0102,
    GORPHO_DIR_Z     = 0x1000,
    GORPHO_DIR_Z_POS = 0x1001,
    GORPHO_DIR_Z_NEG = 0x1002
};

#endif /* __GORPHO_CONSTS_H__ */