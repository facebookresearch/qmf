
TEXFILES := 0_Main.tex 1_Introduction.tex 2_RelatedWork.tex 3_Method.tex 4_Results.tex 5_Conclusion.tex Appendix.tex Commands.tex

paper.pdf: $(TEXFILES)
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make 0_Main.tex
	mv 0_Main.pdf paper.pdf

supp : supplemental.pdf

supplemental.pdf: $(TEXFILES)
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make 0_Supplemental.tex
	mv 0_Supplemental.pdf supplemental.pdf

clean:
	latexmk -CA
