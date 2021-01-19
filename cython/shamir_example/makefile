LIB_DIR = lib


default: example

example: setup.py example.pyx $(LIB_DIR)/libexample.a
	python3 setup.py build_ext --inplace && rm -f example.c && rm -r build

$(LIB_DIR)/libexample.a:
	make -C $(LIB_DIR) libexample.a

clean:
	rm *.so
