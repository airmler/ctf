LOBJS = mapping.o distribution.o topology.o node_aware_dist.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../interface/common.h ../mapping/mapping.h ../mapping/node_aware_dist.h ../shared/util.h ../summation/sum_tsr.h ../tensor/untyped_tensor.h

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

