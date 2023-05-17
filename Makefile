all: run

VERSION:=v0.0.1

push:
	docker push rkalachev/runm_coursework:$(VERSION)

build: requirements.txt Dockerfile
	docker build . -t rkalachev/runm_coursework:$(VERSION)
	touch build

run: build
	docker run -v $(PWD):/usr/src/app rkalachev/runm_coursework:$(VERSION)


doc: svm.pdf 


%.pdf: %.md
	pandoc $^  --toc --pdf-engine=xelatex --template=my_template.tex --variable mainfont="Heuristica" --variable sansfont="Heuristica" --variable mainlang=bulgarian -o $@


.PHONY:  all run push doc
