
??	/host:CPU??@??")???#????"*notTraced-nonXla" "??ǀ????"??ʔ?۟?"??????ʢ"???????"??????"??????"??????"??????"??????"??̠??"??????"??ҡ??"??????"??????"??٣??"?΢???"??????"?䢥??"??????"?휦??"??٦??"?鐧??"??ӧ??"??????"??Ԩ??"??????"??????"??????"??????"?ˋ???a"????????"  " *0"=???????" ?????桁?" " "?????桁?"  "???????"g????"*	 PU_0_bfc"  " ??" ????":$?? ?>" ??W" " ?"	 ????q"  "  "???????"g?ġ?"*	 PU_0_bfc"  " ??" ????"nN뛿?>" ??W" " ?"	 ?Ƅ?q"  "  "???????"??ܦ??0"???????"??????"??????"???????"???????"???????"??????"??????"?ɴ????"  "?ۨ????"  "! "?????"*	 PU_0_bfc"  " ??" ????":$?? ?>" ??W" " ?"	 ????q"
*
LogicalAnd"  "
*output" 
"* ]"  "???????
"??Ɂ???"????ྥ"??փ??"???????"  "??օ??C"  "! "?⚋???"???????"??????"???????"??????"???????"?Ο???"????????""????????"!??????՗"??????"??????Ζ"  " ?????띕"  "! "??????O"??????"????"?ˢ????"??????$"???????"g????"*	 PU_0_bfc"  " ??" ????"nN뛿?>" ??W" " ?"	 ????q"  "  "?ɀ???"??σ"????!???3"%????!???"$????!?ñ"????!??"????!???"  "#?灣!??g"  "! ?"?
tf_Compute??"=???????L" ?????桁?"  " "?????桁?"  ">?ܼ??ϒH"*IteratorGetNext"*
ExpandDims"  "!  "??ڇ?"*	 PU_0_bfc"  " ??" ????"`T?4? ?>" ??W" ?" ?"	 ?Ą?q"
*SameWorkerRecvDone"  "*dynamic" "* 32]"  "6??????a" ?????桁?" "?????桁?"  ">????	ࢺ?" ?????桁?"  " "?????桁?"  "?????	"*	 PU_0_bfc"  " ??<" ?÷?"?cS)܄?>" ??W" ? " ? "	 ?І?q";
*7sequential/dropout/dropout/random_uniform/RandomUniform" ?????桁?"
*output" "* 32"  "?????
"*	 PU_0_bfc"  " ??<" ????".?Z?]?>" ??W" ?" ?"	 ?Ȅ?q"+
*'sequential/dropout/dropout/GreaterEqual" ?????桁?"
*output" 
"* 32"  "n??ɶ
"*	 PU_0_bfc"  " ??<" ?۷?"???-??>" ??W" ? " ? "	 ?І?q"  "  "  "?????
"*	 PU_0_bfc"  " ??<" ????".?Z?]?>" ??W" ? " ? "	 ?І?q"#
*sequential/dropout/dropout/Cast" ?????桁?"
*output" "* 32"  "n????
"*	 PU_0_bfc"  " ??<" ?÷?"?cS)܄?>" ??W" ?" ?"	 ?Ȅ?q"  "  "  "?????
"*	 PU_0_bfc"  " ??<" ????" 79??z?>" ??W" ?" ?"	 ?Ȅ?q"
*sequential/dense/MatMul" ?????桁?"
*output" "* 32"  "?????
"*	 PU_0_bfc"  " ??<" ????"? W?p?>" ??W" ?" ?"	 ?ʄ?q"
*sub" ?????桁?"
*output" "* 32"  "?????
"*	 PU_0_bfc"  " ??<" ????"??<?g?>" ??W" ?" ?"	 ?̄?q"(
*$mean_squared_error/SquaredDifference" ?????桁?"
*output" "* 32"  "n?ҹ?
"*	 PU_0_bfc"  " ??<" ????"? W?p?>" ??W" ?" ?"	 ????q"  "  "  "?????
"* pu_host_bfc"  " ?0"	 ?????"  " ?2" " ?"	 ???? "*
*&gradient_tape/mean_squared_error/Shape" ?????桁?"
*output" "* 2]"  "?ౝ?
"* pu_host_bfc"  " ?2"	 ?????"  " ?2" " ?"	 ???? "+
*'gradient_tape/mean_squared_error/Prod_1" ?????桁?"
*output" "* ]"  "??ї?
"*	 PU_0_bfc"  " ??<" ????"??<?g?>" ??W" " ?"	 ????q"(
*$mean_squared_error/weighted_loss/Sum" ?????桁?"*temp" "* ]"  "n????
"*	 PU_0_bfc"  " ??<" ????"? W?p?>" ??W" ?" ?"	 ?̄?q"  "  "  "?????"*	 PU_0_bfc"  " ??<" ????"??<?g?>" ??W" " ?"	 ?̄?q"
*SameWorkerRecvDone"  "*dynamic" "* 2]"  "?????"* pu_host_bfc"  " ?4"	 ?????"  " ?4" " ?"	 ???? "-
*)gradient_tape/mean_squared_error/floordiv" ?????桁?"
*output" "* ]"  "?????"*	 PU_0_bfc"  " ??<" ????".?Z?]?>" ??W" " ?"	 ?΄?q"
*SameWorkerRecvDone"  "*dynamic" "* ]"  "?????"* pu_host_bfc"  " ?6"	 ?????"  " ?6" " ?"	 ???? "
*Size" ?????桁?"
*output" "* ]"  "?ल?"*	 PU_0_bfc"  " ??<" ????"?fxP%S?>" ??W" " ?"	 ?Є?q"	
*Sum_2" ?????桁?"*temp" "* ]"  "n?֠?"*	 PU_0_bfc"  " ??<" ????".?Z?]?>" ??W" ?" ?"	 ?ʄ?q"  "  "  "m?Ӊ?"*	 PU_0_bfc"  " ??<" ????"??<?g?>" ??W" " ?"	 ?Є?q"  "  "  "???Ͷ"*	 PU_0_bfc"  " ??<" ????".?Z?]?>" ??W" " ?"	 ?ʄ?q"
*SameWorkerRecvDone"  "*dynamic" "* ]"  "?????"*	 PU_0_bfc"  " ??<" ????"?fxP%S?>" ??W" " ?"	 ?Є?q"
*SameWorkerRecvDone"  "*dynamic" "* ]"  "=???????/" ?????桁?"  " "?????桁?"  "?????"*	 PU_0_bfc"  " ??<" ????"?9^?/I?>" ??W" " ?"	 ?҄?q"2
*.gradient_tape/mean_squared_error/DynamicStitch" ?????桁?"
*output" "* 2]"  "m??ɋ"*	 PU_0_bfc"  " ??<" ????"\|3A??>" ??W" " ?"	 ?̄?q"  "  "  "m??܌"*	 PU_0_bfc"  " ??<" ????"?9^?/I?>" ??W" " ?"	 ??q"  "  "  "??侓"* pu_host_bfc"  " ?2"	 ?????"  " ?6" " ?"	 ???? "
*SameWorkerRecvDone"  "*dynamic" "* 2]"  ??2
tf_Compute??"7??ϗ???" ?ܾ??????"  " "?ܾ??????"???????"!  "=???????" ?????桁?"  " "?????桁?"  "1???????" ?????桁?" ????쉙??"  "=???????r" ?????桁?"  " "?????桁?"  "??Д?"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ????q"
*Abs" ?????桁?"
*output" "* 32"  "?????"* pu_host_bfc"  " ?0"	 ?????"  " ?0" " ?"	 ???? "
*strided_slice" ?????桁?"
*output" "* ]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" " ?"	 ?Ɔ?q"
*SameWorkerRecvDone"  "*dynamic" "* ]"  "=???????" ?????桁?"  " "?????桁?"  "??쀠"*	 PU_0_bfc"  " ??" ????"?l?@ٜ?>" ??W" " ?"	 ????q"
*Cast" ?????桁?"
*output" "* ]"  "m?ݺ?"*	 PU_0_bfc"  " ??" ????"	?x?ݜ?>" ??W" " ?"	 ?Ɔ?q"  "  "  "=????	???6" ?????桁?"  " "?????桁?"  ??O
tf_Compute??"1???????" ?ܾ??????" "?ܾ??????"=????	???D" ?????桁?"  " "?????桁?"  "?????	"*	 PU_0_bfc"  " ??;" ????"m???b?>" ??W" ? " ? "	 ????q"
*zeros_like_1" ?????桁?"
*output" "* 32"  ?4?@
tf_Compute??"=???????" ?????桁?"  " "?????桁?"  "S???????"*IteratorGetNext"#*sequential/lstm/PartitionedCall"  "!  "?????"*	 PU_0_bfc"  " ??" ????"Lܸ?Ҝ?>" ??W" ?" ?"	 ?Ȅ?q"
*SameWorkerRecvDone"  "*dynamic" "* 32"  "7?݄????" ?ܾ??????"  " "?ܾ??????"	?????̫"!  "=????	???R" ?????桁?"  " "?????桁?"  "?????	"*	 PU_0_bfc"  " ??;" ????"i?q[	??>" ??W" ? " ? "	 ????q"
*zeros_like_2" ?????桁?"
*output" "* 32"  "=???????F" ?????桁?"  " "?????桁?"  "??퇓"*	 PU_0_bfc"  " ??<" ????"?9^?/I?>" ??W" " ?"	 ??q"

*Cast_1" ?????桁?"
*output" "* ]"  "m????"*	 PU_0_bfc"  " ??<" ????"?fxP%S?>" ??W" " ?"	 ?Є?q"  "  "  "m??Ż"*	 PU_0_bfc"  " ??<" ????"? W?p?>" ??W" " ?"	 ??q"  "  "  "??ާ?"*	 PU_0_bfc"  " ??<" ????"??<?g?>" ??W" " ?"	 ??q"
*div_no_nan_1" ?????桁?"
*output" "* ]"  ">????????" ?????????"  " "?????????"  "??ԯ?"*	 PU_0_bfc"  " ??E" ????"?.???s?>" ??W" ??" ??	"	 ????q"(
*$gradients/transpose_9_grad/transpose" ?????????"
*output" "* 30"  "?????"*	 PU_0_bfc"  " ??M" ?æ?"(:??s?>" ??W" ??" ??"	 ??Ƅq"1
*-gradients/strided_slice_grad/StridedSliceGrad" ?????????"
*output" "* 30"  "p????"*	 PU_0_bfc"  " ??C" ????"?????7?>" ??W" ??" ??	"	 ????q"  "  "  "?????"*	 PU_0_bfc"  " ??C" ????"??bʢ?>" ??W" ?" ?"	 ?Ȅ?q",
*(gradients/CudnnRNN_grad/CudnnRNNBackprop" ?????????"
*output" "* 30"  "?????"*	 PU_0_bfc"  " ??C" ????"???8??>" ??W" ? " ? "	 ????q",
*(gradients/CudnnRNN_grad/CudnnRNNBackprop" ?????????"
*output" "* 1"  "??ߵ?"*	 PU_0_bfc"  " ??D" ?ɯ?"??/?d?>" ??W" ? " ? "	 ?І?q",
*(gradients/CudnnRNN_grad/CudnnRNNBackprop" ?????????"
*output" "* 1"  "?????"*	 PU_0_bfc"  " ??E" ????"???????>" ??W" ??" ??"	 ????q",
*(gradients/CudnnRNN_grad/CudnnRNNBackprop" ?????????"
*output" "
* 4480]"  "???֦"*	 PU_0_bfc"  " ??V" ????"?i?nͬ?>" ??W" ??" ??"	 ??΄q",
*(gradients/CudnnRNN_grad/CudnnRNNBackprop" ?????????"*temp" "* 287232]"  "??Ҳ?"*	 PU_0_bfc"  " ??E" ????"???????>" ??W" ??" ??"	 ??΄q",
*(gradients/CudnnRNN_grad/CudnnRNNBackprop" ?????????"  "  "p?נ?"*	 PU_0_bfc"  " ??=" ????"%???z??>" ??W" ??" ??"	 ??Ƅq"  "  "  "n????"*	 PU_0_bfc"  " ??=" ????"??r??A?>" ??W" ?" ?"	 ?Ȅ?q"  "  "  "n????"*	 PU_0_bfc"  " ??=" ????"?櫲??>" ??W" ? " ? "	 ????q"  "  "  "n?ψ?"*	 PU_0_bfc"  " ??=" ?۶?"4&????>" ??W" ? " ? "	 ?І?q"  "  "  "?????"*	 PU_0_bfc"  " ??=" ?Ӷ?"x_?zX?>" ??W" ?" ?"	 ?Ȅ?q"!
*gradients/split_2_grad/concat" ?????????"
*output" "	* 256]"  "???ۛ"*	 PU_0_bfc"  " ??=" ????"t1%Q???>" ??W" ? " ? "	 ????q"(
*$gradients/transpose_6_grad/transpose" ?????????"
*output" "* 32"  "???Ф"*	 PU_0_bfc"  " ??=" ????"?zH??>" ??W" ? " ? "	 ?І?q"(
*$gradients/transpose_7_grad/transpose" ?????????"
*output" "* 32"  "??"*	 PU_0_bfc"  " ??=" ????"????z?>" ??W" ? " ? "	 ????q"(
*$gradients/transpose_8_grad/transpose" ?????????"
*output" "* 32"  "?????"*	 PU_0_bfc"  " ??>" ?ӵ?"{?[???>" ??W" ? " ? "	 ?Ș?q"(
*$gradients/transpose_5_grad/transpose" ?????????"
*output" "* 32"  "??Ȝ?"*	 PU_0_bfc"  " ??>" ?ϵ?"W_????>" ??W" ?" ?"	 ?Є?q"
*gradients/split_grad/concat" ?????????"
*output" "* 1"  "p????"*	 PU_0_bfc"  " ??=" ?۶?"4&????>" ??W" ??" ??"	 ????q"  "  "  "?????"*	 PU_0_bfc"  " ??>" ?ϵ?"W_????>" ??W" ??" ??"	 ????q"!
*gradients/split_1_grad/concat" ?????????"
*output" "* 32"  "n????"*	 PU_0_bfc"  " ??=" ????"?{
?f?>" ??W" ? " ? "	 ?Ș?q"  "  "  "n????"*	 PU_0_bfc"  " ??=" ????"??}+??>" ??W" ? " ? "	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??=" ????"?(4??>" ??W" ? " ? "	 ?І?q"  "  "  "n?׮?"*	 PU_0_bfc"  " ??=" ?϶?"???<'D?>" ??W" ? " ? "	 ????q"  "  "  "7???????
" ?????????" "?????????"  "n????"*	 PU_0_bfc"  " ??<" ????"? Gf:??>" ??W" ? " ? "	 ????q"  "  "  "p????"*	 PU_0_bfc"  " ??5" ????"=????C?" ??W" ??" ??"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??5" ?Ͼ?"z??????" ??W" ? " ? "	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??4" ????"}?3
??" ??W" ? " ? "	 ????q"  "  "  "p????"*	 PU_0_bfc"  " ??-" ????"Ӧ??H5?" ??W" ??" ??"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??-" ????"?=?"҄?" ??W" ? " ? "	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??," ????"??&_??" ??W" ? " ? "	 ?Ѕ?q"  "  "  "p????"*	 PU_0_bfc"  " ??" ????";?z???" ??W" ??%" ??%"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"??ER?" ??W" ?" ?"	 ?愄q"  "  "  "p????"*	 PU_0_bfc"  " ??" ????"??h;??" ??W" ??" ??"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"???????" ??W" ? " ? "	 ????q"  "  "  "=???????" ?????桁?"  " "?????桁?"  "n????"*	 PU_0_bfc"  " ??" ????"x@????" ??W" ?" ?"	 ?Ȅ?q"  "  "  ?O?
tf_Compute??">?ӧ?????" ?????桁?"  " "?????桁?"  "?????"*	 PU_0_bfc"  " ??" ????":$?? ?>" ??W" " ?"	 ????q"
*Adam/add" ?????桁?"
*output" 	"* ]"  "?????"*	 PU_0_bfc"  " ??" ????"VLn?^?>" ??W" " ?"	 ??q"
*SameWorkerRecvDone"  "*dynamic" "* ]"  "?????"*	 PU_0_bfc"  " ??" ????"j?r?b??>" ??W" " ?"	 ????q"
*Adam/Cast_1" ?????桁?"
*output" "* ]"  "m????"*	 PU_0_bfc"  " ??" ????"`T?4? ?>" ??W" " ?"	 ????q"  "  "  "??Α?"*	 PU_0_bfc"  " ??" ????"j?r?b??>" ??W" " ?"	 ????q"
*Adam/Pow" ?????桁?"
*output" "* ]"  ">?????ʷ?" ????쉙??"  " "????쉙??"  "?????"*	 PU_0_bfc"  " ??" ????"?s??a?c>" ??W" ?" ?"	 ?愄q"
*transpose_0" ????쉙??"
*output" "* 30"  "?????"*	 PU_0_bfc"  " ??" ????"?s??a?c>" ??W" ?" ?"	 ????q"

*concat" ????쉙??"
*output" "	* 256]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ????q"	
*split" ????쉙??"
*output" "* 1"  "???ߣ"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ????q"	
*split" ????쉙??"
*output" "* 1"  "???̤"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ????q"	
*split" ????쉙??"
*output" "* 1"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ????q"	
*split" ????쉙??"
*output" "* 1"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ? " ? "	 ????q"
*split_1" ????쉙??"
*output" "* 32"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ? " ? "	 ?ޅ?q"
*split_1" ????쉙??"
*output" "* 32"  "??ش?"*	 PU_0_bfc"  " ??" ????"  " ??W" ? " ? "	 ????q"
*split_1" ????쉙??"
*output" "* 32"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ? " ? "	 ????q"
*split_1" ????쉙??"
*output" "* 32"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ????q"
*split_2" ????쉙??"
*output" "* 32]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ????q"
*split_2" ????쉙??"
*output" "* 32]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ??q"
*split_2" ????쉙??"
*output" "* 32]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ?Ć?q"
*split_2" ????쉙??"
*output" "* 32]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ?Ȇ?q"
*split_2" ????쉙??"
*output" "* 32]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ?ʆ?q"
*split_2" ????쉙??"
*output" "* 32]"  "??֒?"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ?̆?q"
*split_2" ????쉙??"
*output" "* 32]"  "?????"*	 PU_0_bfc"  " ??" ????"  " ??W" ?" ?"	 ?Ά?q"
*split_2" ????쉙??"
*output" "* 32]"  "n????"*	 PU_0_bfc"  " ??" ????"??e.e??>" ??W" ?" ?"	 ????q"  "  "  "???ц"*	 PU_0_bfc"  " ??" ????"??e.e??>" ??W" ? " ? "	 ?І?q"
*transpose_5" ????쉙??"
*output" "* 32"  "n????"*	 PU_0_bfc"  " ??" ????"?`?? ۨ>" ??W" ? " ? "	 ????q"  "  "  "?????"*	 PU_0_bfc"  " ??" ????"??e.e??>" ??W" ? " ? "	 ????q"
*transpose_6" ????쉙??"
*output" "* 32"  "n??"*	 PU_0_bfc"  " ??" ????"?`?? ۨ>" ??W" ? " ? "	 ?ޅ?q"  "  "  "?????"*	 PU_0_bfc"  " ??" ????"??e.e??>" ??W" ? " ? "	 ?ޅ?q"
*transpose_7" ????쉙??"
*output" "* 32"  "n????"*	 PU_0_bfc"  " ??" ????"?`?? ۨ>" ??W" ? " ? "	 ????q"  "  "  "??⑘"*	 PU_0_bfc"  " ??" ????"??e.e??>" ??W" ? " ? "	 ????q"
*transpose_8" ????쉙??"
*output" "* 32"  "n????"*	 PU_0_bfc"  " ??" ????"?`?? ۨ>" ??W" ? " ? "	 ????q"  "  "  "?????"*	 PU_0_bfc"  " ??" ????"	?x?ݜ?>" ??W" ??" ??"	 ????q"
*
concat_1_0" ????쉙??"
*output" "
* 4480]"  "?????"* pu_host_bfc"  " ?0"	 ?????"  " ?0" ?" ?"	 ???? "
*
concat_1_0" ????쉙??"*temp" "	* 128]"  "?௔?"*	 PU_0_bfc"  " ??" ????"	m?^?>" ??W" ?" ?"	 ????q"
*
concat_1_0" ????쉙??"*temp" "	* 128]"  "?????"* pu_host_bfc"  " ?2"	 ?????"  " ?2" D" ?"	 ???? "
*
concat_1_0" ????쉙??"*temp" "* 68]"  "?????"*	 PU_0_bfc"  " ??" ????"	m?^?>" ??W" D" ?"	 ?Ɔ?q"
*
concat_1_0" ????쉙??"*temp" "* 68]"  "?????"*	 PU_0_bfc"  " ??" ????"	?x?ݜ?>" ??W" D" ?"	 ?Ɔ?q"
*
concat_1_0" ????쉙??"  "  "?????"* pu_host_bfc"  " ?0"	 ?????"  " ?2" D" ?"	 ???? "
*
concat_1_0" ????쉙??"  "  "?????"*	 PU_0_bfc"  " ??" ????"%??ۨ>" ??W" ?" ?"	 ????q"
*
concat_1_0" ????쉙??"  "  "?????"* pu_host_bfc"  " ?."	 ?????"  " ?2" ?" ?"	 ???? "
*
concat_1_0" ????쉙??"  "  "n????"*	 PU_0_bfc"  " ??" ????"?EL*?>" ??W" ?" ?"	 ????q"  "  "  "n?ӵ?"*	 PU_0_bfc"  " ??" ????"7y??QW?>" ??W" ?" ?"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"Rm%y??>" ??W" ?" ?"	 ????q"  "  "  "n?ϩ?"*	 PU_0_bfc"  " ??" ????"H1r^?ӭ>" ??W" ?" ?"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"?0?y?ڸ>" ??W" ? " ? "	 ?І?q"  "  "  "n?أ?"*	 PU_0_bfc"  " ??" ????"? ?f?>" ??W" ? " ? "	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????" q?S?^?>" ??W" ? " ? "	 ?ޅ?q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"???KW?>" ??W" ? " ? "	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"?}]֦?>" ??W" ?" ?"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"??a??>" ??W" ?" ?"	 ????q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"????E?>" ??W" ?" ?"	 ??q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"?D?rq??>" ??W" ?" ?"	 ?Ć?q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"??Cg???>" ??W" ?" ?"	 ?Ȇ?q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"o?[?4?>" ??W" ?" ?"	 ?ʆ?q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"gP??>" ??W" ?" ?"	 ?̆?q"  "  "  "n????"*	 PU_0_bfc"  " ??" ????"נ?ŗ??>" ??W" ?" ?"	 ?Ά?q"  "  "  "??Ռ?"*	 PU_0_bfc"  " ??" ????"lk8A???>" ??W" ??" ??"	 ????q"
*CudnnRNN" ????쉙??"
*output" "* 30"  "?????"*	 PU_0_bfc"  " ??" ????"??v??>" ??W" ? " ? "	 ????q"
*CudnnRNN" ????쉙??"
*output" "* 1"  "?????"*	 PU_0_bfc"  " ??" ????"g?E????>" ??W" ? " ? "	 ?Ѕ?q"
*CudnnRNN" ????쉙??"
*output" "* 1"  "?????"*	 PU_0_bfc"  " ?? " ????"N????>" ??W" ??" ??"	 ????q"
*CudnnRNN" ????쉙??"*temp" "* 287232]"  "?????"*	 PU_0_bfc"  " ??E" ????"ԭ?+??>" ??W" ??%" ??%"	 ????q"
*CudnnRNN" ????쉙??"
*output" "* 153600]"  "???˪"*	 PU_0_bfc"  " ??3" ????"c?b?[?" ??W" ??" ??"	 ????q"
*CudnnRNN" ????쉙??"  "  "???ӳ"*	 PU_0_bfc"  " ??;" ????"W?l?>" ??W" ??" ??"	 ????q"
*transpose_9" ????쉙??"
*output" "* 32"  "7????	???" ????쉙??" "????쉙??"  "n??޺	"*	 PU_0_bfc"  " ??;" ?ø?"???6?>" ??W" ?" ?"	 ?Ȅ?q"  "  "  "=????	???`" ?????桁?"  " "?????桁?"  "?????	"*	 PU_0_bfc"  " ??;" ????"???R?#?>" ??W" ? " ? "	 ????q""
*sequential/dropout/dropout/Mul" ?????桁?"
*output" "* 32"  "=???????" ?????桁?"  " "?????桁?"  "?????"*	 PU_0_bfc"  " ??<" ????"r?a?K5?>" ??W" " ?"	 ?ք?q")
*%gradient_tape/mean_squared_error/Cast" ?????桁?"
*output" "* ]"  "m????"*	 PU_0_bfc"  " ??<" ????"?9^?/I?>" ??W" " ?"	 ?ʄ?q"  "  "  "=???????I" ?????桁?"  " "?????桁?"  "m??Ӟ"*	 PU_0_bfc"  " ??" ????"?[?? ?" ??W" " ?"	 ????q"  "  "  "m????"*	 PU_0_bfc"  " ??" ????"'rȉ%?" ??W" " ?"	 ????q"  "  "  "p????"*	 PU_0_bfc"  " ??" ????"??p??>" ??W" ??" ??"	 ????q"  "  "  "7ु??ñ" ?????桁?" "?????桁?"  ??B
tf_Compute??"=???????	" ?????桁?"  " "?????桁?"  "
???????"  "!  "=???????" ?????桁?"  " "?????桁?"  "?????"* pu_host_bfc"  " ?0"	 ?????"  " ?0" " ?"	 ???? "
*sequential/lstm/Shape" ?????桁?"
*output" "* 3]"  "W?????з"*sequential/lstm/Shape"!*sequential/lstm/strided_slice"  "!  "=???????" ?????桁?"  " "?????桁?"  "h?ҳ?"* pu_host_bfc"  " ?."	 ?????"  " ?0" " ?"	 ???? "  "  "  "??Ϻ?"*	 PU_0_bfc"  " ??" ????"`T?4? ?>" ??W" ? " ? "	 ????q"
*sequential/lstm/zeros" ?????桁?"
*output" "* 32"  ">????????" ?????桁?"  " "?????桁?"  "?????"*	 PU_0_bfc"  " ??<" ????"\|3A??>" ??W" " ?"	 ?Ԅ?q"6
*2mean_squared_error/weighted_loss/num_elements/Cast" ?????桁?"
*output" "* ]"  "m????"*	 PU_0_bfc"  " ??<" ????"?fxP%S?>" ??W" " ?"	 ?΄?q"  "  "  "?????"*	 PU_0_bfc"  " ??<" ????"\|3A??>" ??W" " ?"	 ?̄?q"C
*?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan" ?????桁?"
*output" "* ]"  "m????"*	 PU_0_bfc"  " ??<" ????".?Z?]?>" ??W" " ?"	 ?Ԅ?q"  "  "  "??ˤ?"*	 PU_0_bfc"  " ??<" ????".?Z?]?>" ??W" ?" ?"	 ?ʄ?q"9
*5gradient_tape/mean_squared_error/weighted_loss/Tile_1" ?????桁?"
*output" "* 32]"  "m໸?"*	 PU_0_bfc"  " ??<" ????"??<?g?>" ??W" " ?"	 ?̄?q"  "  "  "m????"*	 PU_0_bfc"  " ??<" ????"? W?p?>" ??W" " ?"	 ????q"  "  "  "h????"* pu_host_bfc"  " ?0"	 ?????"  " ?6" " ?"	 ???? "  "  "  "m????"*	 PU_0_bfc"  " ??<" ????" 79??z?>" ??W" " ?"	 ????q"  "  "  "h????"* pu_host_bfc"  " ?."	 ?????"  " ?6" " ?"	 ???? "  "  "  "m????"*	 PU_0_bfc"  " ??<" ?÷?"?cS)܄?>" ??W" " ?"	 ?ք?q"  "  "  "??ą?"*	 PU_0_bfc"  " ??<" ????" 79??z?>" ??W" " ?"	 ????q"
*
div_no_nan" ?????桁?"
*output" "* ]"  "n????"*	 PU_0_bfc"  " ??<" ?÷?"?cS)܄?>" ??W" ?" ?"	 ?Ą?q"  "  "  "n????"*	 PU_0_bfc"  " ??<" ?ŷ?"^?5?ʎ?>" ??W" ?" ?"	 ?Ȅ?q"  "  "  "?????"*	 PU_0_bfc"  " ??<" ????"?u?????>" ??W" ? " ? "	 ????q")
*%gradient_tape/sequential/dense/MatMul" ?????桁?"
*output" "* 32"  "?????"*	 PU_0_bfc"  " ??<" ????"o?? ???>" ??W" ?" ?"	 ?Ą?q"+
*'gradient_tape/sequential/dense/MatMul_1" ?????桁?"
*output" "* 32"  "n??Ġ"*	 PU_0_bfc"  " ??<" ?÷?"?cS)܄?>" ??W" ? " ? "	 ????q"  "  "  "??ڠ?"*	 PU_0_bfc"  " ??<" ????" 79??z?>" ??W" " ?"	 ?Ȅ?q"6
*2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad" ?????桁?"
*output" "* 1]"  "nଯ?"*	 PU_0_bfc"  " ??<" ?÷?"?cS)܄?>" ??W" ?" ?"	 ?ʄ?q"  "  "  "n????"*	 PU_0_bfc"  " ??;" ????"???R?#?>" ??W" ? " ? "	 ?І?q"  "  "  "n?ŷ?"*	 PU_0_bfc"  " ??;" ????"ؾ???-?>" ??W" ?" ?"	 ?Ą?q"  "  "  "m????"*	 PU_0_bfc"  " ??;" ????"L??o?7?>" ??W" " ?"	 ?Ȅ?q"  "  "  "1???????" ?????桁?" ?????????"  "=???????" ?????桁?"  " "?????桁?"  "n????"*	 PU_0_bfc"  " ??" ????"݇???" ??W" ?" ?"	 ?Є?q"  "  "  ??$tf_GPU_Event_Manager??"b??ȗ"* pu_host_bfc"  " ?."	 ?????"  " ?0" " ?"	 ???? "  "  "i????"* pu_host_bfc"  " ?4"	 ?????"???v)0?" ?6" " ?"	 ???? "  "  "i????"* pu_host_bfc"  " ?2"	 ?????"?4??'@?" ?6" " ?"	 ???? "  "  "b൶?"* pu_host_bfc"  " ?0"	 ?????"  " ?6" " ?"	 ???? "  "  "g????"*	 PU_0_bfc"  " ??<" ????"??<?g?>" ??W" " ?"	 ?҄?q"  "  ??tf_data_iterator_get_next??"((???????" ?????桁?"  "  "!'???????" ԥ??͍??x"  "I&????଱" 󰹮ï??" ԥ??͍??x"*true"*true" "  ??<tf_data_iterator_resource??").???????" ??????Ԣ" 󰹮ï??"*+???????" ʓ??ׅ?ԋ" ??????Ԣ"+*???????" ?????????" ʓ??ׅ?ԋ"+)????ๆ" ܴ???????" ?????????")-????ૢ" ????????a" ??????Ԣ"),??????C" ?ߋ??ݡ??" ????????a"*,???????" ??Á?ݡ??" ????????a"/??ً???" ?ܾ??????"(%$% EagerExecute: FlushSummaryWriter"ExecutorDoneCallback"''Iterator::Model"%!GatherV2_1:GatherV2"GatherV2"'!#!EagerLocalExecute: WriteSummary"A#=#%FlushSummaryWriter:FlushSummaryWriter"FlushSummaryWriter"'#IteratorGetNext/_1:_Send"_Send"V-R-5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat"Iterator::ForeverRepeat"51-TensorHandle::GetResourceHandleInfo WaitReady"./*/&InstantiatedCapturedFunction::RunAsync"%!ValidateInputTypeAndPlacement"n)j)OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice"Iterator::TensorSlice"'#IteratorGetNext/_7:_Send"_Send"J+F+/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"Iterator::FlatMap"8
4
IteratorGetNext:IteratorGetNext"IteratorGetNext"train_function"tf.constant"EagerKernelExecute"51-EagerExecute: __inference_train_function_3168" TFE_Py_FastPathExecute_C"#EagerLocalExecute: Identity"wso_Send input 1 from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0"%!EagerLocalExecute: LogicalAnd"MemoryDeallocation"-$)$%EagerLocalExecute: FlushSummaryWriter"PartitionedCallOp"?&;&Iterator::Model::ParallelMapV2"Iterator::ParallelMapV2"MemoryAllocation"/ + WriteSummary:WriteSummary"WriteSummary"ExecutorState::Process""""EagerExecute: WriteSummary"'#EagerCopyToDeviceAndAddCacheKey"_,[,AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor"Iterator::FromTensor"FunctionRun":.6.#Iterator::Model::ParallelMapV2::Zip"Iterator::Zip"#		GatherV2:GatherV2"GatherV2"$( (IteratorGetNextOp::DoCompute"51"sequential/lstm/Shape/_3:_HostSend"	_HostSend"^*Z*?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"Iterator::Concatenate"TFE_DeleteTensorHandle"wso_Send input 0 from /job:localhost/replica:0/task:0/device:CPU:0 to /job:localhost/replica:0/task:0/device:GPU:0":62EagerLocalExecute: __inference_train_function_3168" TFE_Py_ExecuteCancelable"#Identity:Identity"Identity"EagerExecute: Identity")%LogicalAnd:LogicalAnd"
LogicalAnd" EagerExecute: LogicalAnd*bytes_available*peak_bytes_in_use*!!is_eager*bytes_reserved*region_type*	shape*allocation_bytes*
	
tf_op*parent_step_id*
_r*
to*_ct*autotune*parallelism*
_p*tracing_count*fragmentation*requested_bytes*allocator_name*  	step_name*bytes_allocated*	data_type*from*		addr*
id*function_step_id*_pt*iter_num*
_c*deterministic*group_id*tf_function_call*	parent_id
?/device:GPU:0*compute_cap_major*memory_size*group_id*compute_cap_minor*
clock_rate*memory_bandwidth*
core_count*		is_eager*	step_name2 ??k2 .2	??ȕ?2????2 2 @JONG8: Failed to load libcupti (is it installed and accessible?)