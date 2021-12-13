DOXYGEN=doxygen

doc: clean copy_footer generate_doc

copy_footer:
	cp footer/*.htm .
	
copy_footer_sf:
	cp footer_sf/*.htm .	
	
generate_doc:
	$(DOXYGEN) overview.dox
	$(DOXYGEN) reference.dox
	$(DOXYGEN) tutorial.dox
	$(DOXYGEN) examples.dox
	$(DOXYGEN) design.dox

clean:
	rm -rf ../../../doc
	rm -rf ../../../threadpool-$(RELEASE_VERSION)-doc
	rm -rf ../../../threadpool-$(RELEASE_VERSION)-doc.zip

release_doc: clean doc
	cd ../../..; mv doc threadpool-$(RELEASE_VERSION)-doc
	cd ../../..; zip -r9 threadpool-$(RELEASE_VERSION)-doc.zip threadpool-$(RELEASE_VERSION)-doc
	cd ../../..; mv threadpool-$(RELEASE_VERSION)-doc doc

doc_sf: clean copy_footer_sf generate_doc

deploy_sf: doc_sf
	scp -vr ../../../doc/*  $(RELEASE_MANAGER)@shell.sourceforge.net:/home/groups/t/th/threadpool/htdocs	
	




