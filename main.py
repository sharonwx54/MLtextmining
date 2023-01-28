import main_funcs as mf
import sys
# NOTE running main assume all pickle files that are preprocessed exists
# if not, need to call preprocess first

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # with argument, we call svm or rmlr with ctf or df features
        run_type = sys.argv[1]
    else:
        # without argument, we call rmlr with engineered feature
        run_type = ""
    mf.preprocess_main()
    mf.main(run_type)
