Search.setIndex({envversion:47,filenames:["index","modules","nbodykit","nbodykit.corrfrompower","nbodykit.dataset","nbodykit.distributedarray","nbodykit.extensionpoints","nbodykit.files","nbodykit.fof","nbodykit.halos","nbodykit.measurestats","nbodykit.ndarray","nbodykit.pluginmanager","nbodykit.plugins","nbodykit.stripedfile","nbodykit.utils","nbodykit.utils.selectionlanguage","nbodykit.utils.taskmanager","plugins/modules","plugins/plugins","plugins/plugins.algorithms","plugins/plugins.algorithms.ExampleAlgorithm","plugins/plugins.algorithms.FOF","plugins/plugins.algorithms.FOF6D","plugins/plugins.algorithms.PairCountCorrelation","plugins/plugins.algorithms.PeriodicBox","plugins/plugins.algorithms.Subsample","plugins/plugins.algorithms.TidalTensor","plugins/plugins.algorithms.TraceHalo","plugins/plugins.datasource","plugins/plugins.datasource.FOFGroups","plugins/plugins.datasource.FastPM","plugins/plugins.datasource.Gadget","plugins/plugins.datasource.Grid","plugins/plugins.datasource.HaloLabel","plugins/plugins.datasource.Pandas","plugins/plugins.datasource.PlainText","plugins/plugins.datasource.TPMLabel","plugins/plugins.datasource.TPMSnapshot","plugins/plugins.painter","plugins/plugins.painter.DefaultPainter","plugins/plugins.painter.GridPainter","plugins/plugins.storage","plugins/plugins.storage.Measurement1DStorage","plugins/plugins.storage.Measurement2DStorage","plugins/plugins.transfer","plugins/plugins.transfer.TransferFunction"],objects:{"":{nbodykit:[2,8,0,"-"],plugins:[19,8,0,"-"]},"nbodykit.corrfrompower":{corrfrompower:[3,9,1,""],interp1d:[3,10,1,""]},"nbodykit.dataset":{Corr1dDataSet:[4,10,1,""],Corr2dDataSet:[4,10,1,""],DataSet:[4,10,1,""],Power1dDataSet:[4,10,1,""],Power2dDataSet:[4,10,1,""],bin_ndarray:[4,9,1,""],from_1d_measurement:[4,9,1,""],from_2d_measurement:[4,9,1,""]},"nbodykit.dataset.Corr1dDataSet":{from_nbkit:[4,11,1,""]},"nbodykit.dataset.Corr2dDataSet":{from_nbkit:[4,11,1,""]},"nbodykit.dataset.DataSet":{average:[4,12,1,""],copy:[4,12,1,""],from_nbkit:[4,11,1,""],reindex:[4,12,1,""],rename_variable:[4,12,1,""],sel:[4,12,1,""],shape:[4,13,1,""],squeeze:[4,12,1,""],variables:[4,13,1,""]},"nbodykit.dataset.Power1dDataSet":{from_nbkit:[4,11,1,""]},"nbodykit.dataset.Power2dDataSet":{from_nbkit:[4,11,1,""]},"nbodykit.distributedarray":{DistributedArray:[5,10,1,""],EmptyRankType:[5,10,1,""],LinearTopology:[5,10,1,""],test:[5,9,1,""]},"nbodykit.distributedarray.DistributedArray":{bincount:[5,12,1,""],sort:[5,12,1,""],unique_labels:[5,12,1,""]},"nbodykit.distributedarray.LinearTopology":{heads:[5,12,1,""],next:[5,12,1,""],prev:[5,12,1,""],tails:[5,12,1,""]},"nbodykit.extensionpoints":{Algorithm:[6,10,1,""],DataSource:[6,10,1,""],MeasurementStorage:[6,10,1,""],Painter:[6,10,1,""],Transfer:[6,10,1,""]},"nbodykit.extensionpoints.Algorithm":{help_formatter:[6,13,1,""],parse_known_yaml:[6,11,1,""],plugins:[6,13,1,""],registry:[6,13,1,""],run:[6,12,1,""],save:[6,12,1,""]},"nbodykit.extensionpoints.DataSource":{BoxSizeParser:[6,14,1,""],fromstring:[6,11,1,""],help_formatter:[6,13,1,""],plugins:[6,13,1,""],read:[6,12,1,""],readall:[6,12,1,""],registry:[6,13,1,""]},"nbodykit.extensionpoints.MeasurementStorage":{"new":[6,11,1,""],add_storage_klass:[6,11,1,""],help_formatter:[6,13,1,""],klasses:[6,13,1,""],open:[6,12,1,""],plugin_name:[6,13,1,""],plugins:[6,13,1,""],registry:[6,13,1,""],write:[6,12,1,""]},"nbodykit.extensionpoints.Painter":{fromstring:[6,11,1,""],help_formatter:[6,13,1,""],paint:[6,12,1,""],plugins:[6,13,1,""],read_and_decompose:[6,12,1,""],registry:[6,13,1,""]},"nbodykit.extensionpoints.Transfer":{fromstring:[6,11,1,""],help_formatter:[6,13,1,""],plugins:[6,13,1,""],registry:[6,13,1,""]},"nbodykit.files":{TPMSnapshotFile:[7,10,1,""]},"nbodykit.files.TPMSnapshotFile":{create:[7,11,1,""],read:[7,12,1,""],write:[7,12,1,""]},"nbodykit.fof":{assign_halo_label:[8,9,1,""],fof:[8,9,1,""],local_fof:[8,9,1,""],split_size_2d:[8,9,1,""]},"nbodykit.halos":{centerofmass:[9,9,1,""],count:[9,9,1,""]},"nbodykit.measurestats":{compute_3d_corr:[10,9,1,""],compute_3d_power:[10,9,1,""],compute_brutal_corr:[10,9,1,""],project_to_basis:[10,9,1,""]},"nbodykit.ndarray":{equiv_class:[11,9,1,""],replacesorted:[11,9,1,""]},"nbodykit.pluginmanager":{ArgumentParser:[12,10,1,""],HelpFormatterColon:[12,10,1,""],ListPluginsAction:[12,9,1,""],add_plugin_list_argument:[12,9,1,""],load:[12,9,1,""]},"nbodykit.pluginmanager.ArgumentParser":{convert_args_file_to_args:[12,12,1,""]},"nbodykit.utils":{selectionlanguage:[16,8,0,"-"],taskmanager:[17,8,0,"-"]},"nbodykit.utils.selectionlanguage":{And:[16,10,1,""],CaselessKeyword:[16,10,1,""],CaselessLiteral:[16,10,1,""],CharsNotIn:[16,10,1,""],Combine:[16,10,1,""],Dict:[16,10,1,""],Each:[16,10,1,""],Empty:[16,10,1,""],FollowedBy:[16,10,1,""],Forward:[16,10,1,""],GoToColumn:[16,10,1,""],Group:[16,10,1,""],Keyword:[16,10,1,""],LineEnd:[16,10,1,""],LineStart:[16,10,1,""],Literal:[16,10,1,""],MatchFirst:[16,10,1,""],NoMatch:[16,10,1,""],NotAny:[16,10,1,""],OneOrMore:[16,10,1,""],OnlyOnce:[16,10,1,""],Optional:[16,10,1,""],Or:[16,10,1,""],ParseBaseException:[16,15,1,""],ParseElementEnhance:[16,10,1,""],ParseException:[16,15,1,""],ParseExpression:[16,10,1,""],ParseFatalException:[16,15,1,""],ParseResults:[16,10,1,""],ParseSyntaxException:[16,15,1,""],ParserElement:[16,10,1,""],QuotedString:[16,10,1,""],RecursiveGrammarException:[16,15,1,""],Regex:[16,10,1,""],SkipTo:[16,10,1,""],StringEnd:[16,10,1,""],StringStart:[16,10,1,""],Suppress:[16,10,1,""],Token:[16,10,1,""],TokenConverter:[16,10,1,""],Upcase:[16,10,1,""],White:[16,10,1,""],Word:[16,10,1,""],WordEnd:[16,10,1,""],WordStart:[16,10,1,""],ZeroOrMore:[16,10,1,""],col:[16,9,1,""],countedArray:[16,9,1,""],delimitedList:[16,9,1,""],dictOf:[16,9,1,""],downcaseTokens:[16,9,1,""],indentedBlock:[16,9,1,""],infixNotation:[16,9,1,""],keepOriginalText:[16,9,1,""],line:[16,9,1,""],lineno:[16,9,1,""],locatedExpr:[16,9,1,""],makeHTMLTags:[16,9,1,""],makeXMLTags:[16,9,1,""],matchOnlyAtCol:[16,9,1,""],matchPreviousExpr:[16,9,1,""],matchPreviousLiteral:[16,9,1,""],nestedExpr:[16,9,1,""],nullDebugAction:[16,9,1,""],oneOf:[16,9,1,""],operatorPrecedence:[16,9,1,""],originalTextFor:[16,9,1,""],removeQuotes:[16,9,1,""],replaceHTMLEntity:[16,9,1,""],replaceWith:[16,9,1,""],srange:[16,9,1,""],traceParseAction:[16,9,1,""],ungroup:[16,9,1,""],upcaseTokens:[16,9,1,""],withAttribute:[16,9,1,""]},"nbodykit.utils.selectionlanguage.And":{checkRecursion:[16,12,1,""],parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.CaselessKeyword":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.CaselessLiteral":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.CharsNotIn":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Combine":{ignore:[16,12,1,""],postParse:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Dict":{postParse:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Each":{checkRecursion:[16,12,1,""],parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.FollowedBy":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Forward":{copy:[16,12,1,""],leaveWhitespace:[16,12,1,""],streamline:[16,12,1,""],validate:[16,12,1,""]},"nbodykit.utils.selectionlanguage.GoToColumn":{parseImpl:[16,12,1,""],preParse:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Group":{postParse:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Keyword":{DEFAULT_KEYWORD_CHARS:[16,13,1,""],copy:[16,12,1,""],parseImpl:[16,12,1,""],setDefaultKeywordChars:[16,14,1,""]},"nbodykit.utils.selectionlanguage.LineEnd":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.LineStart":{parseImpl:[16,12,1,""],preParse:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Literal":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.MatchFirst":{checkRecursion:[16,12,1,""],parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.NoMatch":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.NotAny":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.OneOrMore":{parseImpl:[16,12,1,""],setResultsName:[16,12,1,""]},"nbodykit.utils.selectionlanguage.OnlyOnce":{reset:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Optional":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Or":{checkRecursion:[16,12,1,""],parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.ParseBaseException":{markInputline:[16,12,1,""]},"nbodykit.utils.selectionlanguage.ParseElementEnhance":{checkRecursion:[16,12,1,""],ignore:[16,12,1,""],leaveWhitespace:[16,12,1,""],parseImpl:[16,12,1,""],streamline:[16,12,1,""],validate:[16,12,1,""]},"nbodykit.utils.selectionlanguage.ParseExpression":{append:[16,12,1,""],copy:[16,12,1,""],ignore:[16,12,1,""],leaveWhitespace:[16,12,1,""],setResultsName:[16,12,1,""],streamline:[16,12,1,""],validate:[16,12,1,""]},"nbodykit.utils.selectionlanguage.ParseResults":{append:[16,12,1,""],asDict:[16,12,1,""],asList:[16,12,1,""],asXML:[16,12,1,""],clear:[16,12,1,""],copy:[16,12,1,""],dump:[16,12,1,""],extend:[16,12,1,""],get:[16,12,1,""],getName:[16,12,1,""],haskeys:[16,12,1,""],insert:[16,12,1,""],items:[16,12,1,""],iteritems:[16,12,1,""],iterkeys:[16,12,1,""],itervalues:[16,12,1,""],keys:[16,12,1,""],pop:[16,12,1,""],pprint:[16,12,1,""],values:[16,12,1,""]},"nbodykit.utils.selectionlanguage.ParserElement":{DEFAULT_WHITE_CHARS:[16,13,1,""],addParseAction:[16,12,1,""],checkRecursion:[16,12,1,""],copy:[16,12,1,""],enablePackrat:[16,14,1,""],ignore:[16,12,1,""],inlineLiteralsUsing:[16,14,1,""],leaveWhitespace:[16,12,1,""],literalStringClass:[16,13,1,""],parseFile:[16,12,1,""],parseImpl:[16,12,1,""],parseString:[16,12,1,""],parseWithTabs:[16,12,1,""],postParse:[16,12,1,""],preParse:[16,12,1,""],resetCache:[16,14,1,""],scanString:[16,12,1,""],searchString:[16,12,1,""],setBreak:[16,12,1,""],setDebug:[16,12,1,""],setDebugActions:[16,12,1,""],setDefaultWhitespaceChars:[16,14,1,""],setFailAction:[16,12,1,""],setName:[16,12,1,""],setParseAction:[16,12,1,""],setResultsName:[16,12,1,""],setWhitespaceChars:[16,12,1,""],streamline:[16,12,1,""],suppress:[16,12,1,""],transformString:[16,12,1,""],tryParse:[16,12,1,""],validate:[16,12,1,""],verbose_stacktrace:[16,13,1,""]},"nbodykit.utils.selectionlanguage.QuotedString":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Regex":{compiledREtype:[16,13,1,""],parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.SkipTo":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.StringEnd":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.StringStart":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Suppress":{postParse:[16,12,1,""],suppress:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Token":{setName:[16,12,1,""]},"nbodykit.utils.selectionlanguage.Upcase":{postParse:[16,12,1,""]},"nbodykit.utils.selectionlanguage.White":{parseImpl:[16,12,1,""],whiteStrs:[16,13,1,""]},"nbodykit.utils.selectionlanguage.Word":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.WordEnd":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.WordStart":{parseImpl:[16,12,1,""]},"nbodykit.utils.selectionlanguage.ZeroOrMore":{parseImpl:[16,12,1,""],setResultsName:[16,12,1,""]},"nbodykit.utils.taskmanager":{"enum":[17,9,1,""],TaskManager:[17,10,1,""],replacements_from_file:[17,9,1,""],split_ranks:[17,9,1,""],tasks_parser:[17,9,1,""]},"nbodykit.utils.taskmanager.TaskManager":{create:[17,11,1,""],parse_args:[17,11,1,""],run_all:[17,12,1,""]},"plugins.algorithms":{ExampleAlgorithm:[21,8,0,"-"],FOF6D:[23,8,0,"-"],FOF:[22,8,0,"-"],PairCountCorrelation:[24,8,0,"-"],PeriodicBox:[25,8,0,"-"],Subsample:[26,8,0,"-"],TidalTensor:[27,8,0,"-"],TraceHalo:[28,8,0,"-"]},"plugins.algorithms.ExampleAlgorithm":{Describe:[21,10,1,""]},"plugins.algorithms.ExampleAlgorithm.Describe":{finalize_attributes:[21,12,1,""],logger:[21,13,1,""],plugin_name:[21,13,1,""],register:[21,11,1,""],run:[21,12,1,""],save:[21,12,1,""]},"plugins.algorithms.FOF":{FOFAlgorithm:[22,10,1,""]},"plugins.algorithms.FOF.FOFAlgorithm":{plugin_name:[22,13,1,""],register:[22,11,1,""],run:[22,12,1,""],save:[22,12,1,""]},"plugins.algorithms.FOF6D":{FOF6DAlgorithm:[23,10,1,""],so:[23,9,1,""],subfof:[23,9,1,""]},"plugins.algorithms.FOF6D.FOF6DAlgorithm":{plugin_name:[23,13,1,""],register:[23,11,1,""],run:[23,12,1,""],save:[23,12,1,""]},"plugins.algorithms.PairCountCorrelation":{PairCountCorrelationAlgorithm:[24,10,1,""],binning_type:[24,9,1,""]},"plugins.algorithms.PairCountCorrelation.PairCountCorrelationAlgorithm":{plugin_name:[24,13,1,""],register:[24,11,1,""],run:[24,12,1,""],save:[24,12,1,""]},"plugins.algorithms.PeriodicBox":{FFTCorrelationAlgorithm:[25,10,1,""],FFTPowerAlgorithm:[25,10,1,""],FieldsType:[25,9,1,""]},"plugins.algorithms.PeriodicBox.FFTCorrelationAlgorithm":{logger:[25,13,1,""],plugin_name:[25,13,1,""],register:[25,11,1,""],run:[25,12,1,""],save:[25,12,1,""]},"plugins.algorithms.PeriodicBox.FFTPowerAlgorithm":{logger:[25,13,1,""],plugin_name:[25,13,1,""],register:[25,11,1,""],run:[25,12,1,""],save:[25,12,1,""]},"plugins.algorithms.Subsample":{Subsample:[26,10,1,""]},"plugins.algorithms.Subsample.Subsample":{plugin_name:[26,13,1,""],register:[26,11,1,""],run:[26,12,1,""],save:[26,12,1,""],write_hdf5:[26,12,1,""],write_mwhite_subsample:[26,12,1,""]},"plugins.algorithms.TidalTensor":{TidalTensor:[27,10,1,""]},"plugins.algorithms.TidalTensor.TidalTensor":{NormalizeDC:[27,12,1,""],Smoothing:[27,12,1,""],TidalTensor:[27,12,1,""],plugin_name:[27,13,1,""],register:[27,11,1,""],run:[27,12,1,""],save:[27,12,1,""],write_hdf5:[27,12,1,""]},"plugins.algorithms.TraceHalo":{TraceHaloAlgorithm:[28,10,1,""]},"plugins.algorithms.TraceHalo.TraceHaloAlgorithm":{plugin_name:[28,13,1,""],register:[28,11,1,""],run:[28,12,1,""],save:[28,12,1,""]},"plugins.datasource":{FOFGroups:[30,8,0,"-"],FastPM:[31,8,0,"-"],Gadget:[32,8,0,"-"],Grid:[33,8,0,"-"],HaloLabel:[34,8,0,"-"],Pandas:[35,8,0,"-"],PlainText:[36,8,0,"-"],TPMLabel:[37,8,0,"-"],TPMSnapshot:[38,8,0,"-"]},"plugins.datasource.FOFGroups":{FOFDataSource:[30,10,1,""]},"plugins.datasource.FOFGroups.FOFDataSource":{finalize_attributes:[30,12,1,""],plugin_name:[30,13,1,""],readall:[30,12,1,""],register:[30,11,1,""]},"plugins.datasource.FastPM":{FastPMDataSource:[31,10,1,""]},"plugins.datasource.FastPM.FastPMDataSource":{finalize_attributes:[31,12,1,""],plugin_name:[31,13,1,""],read:[31,12,1,""],register:[31,11,1,""]},"plugins.datasource.Gadget":{GadgetDataSource:[32,10,1,""],GadgetGroupTabDataSource:[32,10,1,""]},"plugins.datasource.Gadget.GadgetDataSource":{plugin_name:[32,13,1,""],read:[32,12,1,""],register:[32,11,1,""]},"plugins.datasource.Gadget.GadgetGroupTabDataSource":{plugin_name:[32,13,1,""],read:[32,12,1,""],register:[32,11,1,""]},"plugins.datasource.Grid":{GridDataSource:[33,10,1,""]},"plugins.datasource.Grid.GridDataSource":{plugin_name:[33,13,1,""],register:[33,11,1,""]},"plugins.datasource.HaloLabel":{HaloLabel:[34,10,1,""]},"plugins.datasource.HaloLabel.HaloLabel":{plugin_name:[34,13,1,""],read:[34,12,1,""],register:[34,11,1,""]},"plugins.datasource.Pandas":{PandasDataSource:[35,10,1,""],list_str:[35,9,1,""]},"plugins.datasource.Pandas.PandasDataSource":{plugin_name:[35,13,1,""],readall:[35,12,1,""],register:[35,11,1,""]},"plugins.datasource.PlainText":{PlainTextDataSource:[36,10,1,""],list_str:[36,9,1,""]},"plugins.datasource.PlainText.PlainTextDataSource":{plugin_name:[36,13,1,""],readall:[36,12,1,""],register:[36,11,1,""]},"plugins.datasource.TPMLabel":{TPMLabel:[37,10,1,""]},"plugins.datasource.TPMLabel.TPMLabel":{plugin_name:[37,13,1,""],read:[37,12,1,""],register:[37,11,1,""]},"plugins.datasource.TPMSnapshot":{TPMSnapshotDataSource:[38,10,1,""]},"plugins.datasource.TPMSnapshot.TPMSnapshotDataSource":{plugin_name:[38,13,1,""],read:[38,12,1,""],register:[38,11,1,""]},"plugins.painter":{DefaultPainter:[40,8,0,"-"],GridPainter:[41,8,0,"-"]},"plugins.painter.DefaultPainter":{DefaultPainter:[40,10,1,""]},"plugins.painter.DefaultPainter.DefaultPainter":{paint:[40,12,1,""],plugin_name:[40,13,1,""],register:[40,11,1,""]},"plugins.painter.GridPainter":{GridPainter:[41,10,1,""]},"plugins.painter.GridPainter.GridPainter":{paint:[41,12,1,""],plugin_name:[41,13,1,""],register:[41,11,1,""]},"plugins.storage":{Measurement1DStorage:[43,8,0,"-"],Measurement2DStorage:[44,8,0,"-"]},"plugins.storage.Measurement1DStorage":{Measurement1DStorage:[43,10,1,""]},"plugins.storage.Measurement1DStorage.Measurement1DStorage":{plugin_name:[43,13,1,""],register:[43,11,1,""],write:[43,12,1,""]},"plugins.storage.Measurement2DStorage":{Measurement2DStorage:[44,10,1,""]},"plugins.storage.Measurement2DStorage.Measurement2DStorage":{plugin_name:[44,13,1,""],register:[44,11,1,""],write:[44,12,1,""]},"plugins.transfer":{TransferFunction:[46,8,0,"-"]},"plugins.transfer.TransferFunction":{AnisotropicCIC:[46,10,1,""],NormalizeDC:[46,10,1,""],RemoveDC:[46,10,1,""]},"plugins.transfer.TransferFunction.AnisotropicCIC":{plugin_name:[46,13,1,""],register:[46,11,1,""]},"plugins.transfer.TransferFunction.NormalizeDC":{plugin_name:[46,13,1,""],register:[46,11,1,""]},"plugins.transfer.TransferFunction.RemoveDC":{plugin_name:[46,13,1,""],register:[46,11,1,""]},nbodykit:{corrfrompower:[3,8,0,"-"],dataset:[4,8,0,"-"],distributedarray:[5,8,0,"-"],extensionpoints:[6,8,0,"-"],files:[7,8,0,"-"],fof:[8,8,0,"-"],halos:[9,8,0,"-"],measurestats:[10,8,0,"-"],ndarray:[11,8,0,"-"],pluginmanager:[12,8,0,"-"],plugins:[13,8,0,"-"],stripedfile:[14,8,0,"-"],utils:[15,8,0,"-"]},plugins:{algorithms:[20,8,0,"-"],datasource:[29,8,0,"-"],painter:[39,8,0,"-"],storage:[42,8,0,"-"],transfer:[45,8,0,"-"]}},objnames:{"0":["np","module","Python module"],"1":["np","function","Python function"],"10":["py","class","Python class"],"11":["py","classmethod","Python class method"],"12":["py","method","Python method"],"13":["py","attribute","Python attribute"],"14":["py","staticmethod","Python static method"],"15":["py","exception","Python exception"],"2":["np","class","Python class"],"3":["np","classmethod","Python class method"],"4":["np","method","Python method"],"5":["np","attribute","Python attribute"],"6":["np","staticmethod","Python static method"],"7":["np","exception","Python exception"],"8":["py","module","Python module"],"9":["py","function","Python function"]},objtypes:{"0":"np:module","1":"np:function","10":"py:class","11":"py:classmethod","12":"py:method","13":"py:attribute","14":"py:staticmethod","15":"py:exception","2":"np:class","3":"np:classmethod","4":"np:method","5":"np:attribute","6":"np:staticmethod","7":"np:exception","8":"py:module","9":"py:function"},terms:{"0123456789abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz":16,"1e14":[30,35,36],"1st":10,"2nd":10,"95eab740e1b08b78c03f":4,"__call__":[6,16],"_cen":4,"_errorstop":16,"_fields_to_sum":4,"_nulltoken":16,"_positiontoken":16,"abstract":16,"boolean":[4,5,8,11,16],"break":16,"case":[10,16,17],"char":16,"class":[3,4,5,6,7,8,11,12,16,17,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,43,44,46],"default":[6,8,10,16,43],"enum":17,"float":[4,6,8,9,10,30,33,35,36],"function":[3,4,6,10,12,16,24,25,46],"import":[4,16],"int":[8,10,17,40,41],"long":16,"new":[4,5,6,8,12,16],"null":16,"return":[3,4,5,6,8,9,10,11,12,16,17,24,25,40,41],"static":[6,16],"throw":16,"true":[3,4,5,6,8,10,11,16,32,37,38],"try":[6,17],"while":[10,16],abbrevi:16,abcdefghijklmnopqrstuvwxyz:16,abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz0123456789_:16,about:[10,16],abov:16,accept:[12,16],access:[4,16],accord:16,account:46,action:[12,16],activ:16,acton:16,actual:[3,16],adapt:8,add:[6,12,16],add_argu:6,add_plugin_list_argu:12,add_storage_klass:6,addit:[4,16,43,44],addparseact:16,addresse:16,adjac:16,advanc:16,aeioui:16,after:[4,5,9,16],aka:3,algebra:16,algorithm:[0,6,8,17,18,19],algorithm_nam:17,alia:[4,6,16],align:16,all:[4,5,6,8,9,16,17,35,36],allow:[16,43],along:[4,5],alpha:16,alphanum:16,alphanumer:16,alreadi:11,also:[3,4,8,10,12,16],altern:16,alwai:[3,16],amplitud:[27,46],angl:10,ani:[4,8,16,43,44],anisotropicc:[6,46],anoth:16,any_valu:16,api:0,append:[4,16],appli:[6,11],approach:16,appropri:41,arang:11,area:10,arg:[6,7,12,16],argpars:12,args_fil:12,argument:[4,6,12,16,17,24,25,35,43],argumentpars:12,arr:11,arrai:[4,5,6,8,10,11,16,24,30,35,36,43,44],array_lik:[4,5,6,8,9,10,11,24,30,35,36,43],ascii:36,asdict:16,askeyword:16,aslist:16,assign:[5,8,16],assign_halo_label:8,associ:[16,24],asstr:16,assum:[3,9,10,35,36],asxml:16,atom:16,attach:[4,16],attempt:[16,17],attr:4,attrdict:16,attribut:[4,5,6,16],avail:17,averag:4,avoid:16,axi:4,back:16,backslash:16,backward:16,base:[3,4,5,6,7,12,16,17,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,43,44,46],baseexpr:16,basenam:7,basi:10,basic:16,becaus:[12,16],been:[4,8,16],befor:[4,5,12,16],begin:[16,35,36,43],behavior:[6,16,30,35,36],below:[4,16],best:16,between:16,beyond:16,bin:[4,10,24,25,43,44],bin_ndarrai:4,binari:[16,33,41],bincount:5,binning_typ:24,block:16,blockstat:16,blockstatementexpr:16,bodi:16,bodychar:16,bool:[4,10],borrow:16,both:[10,12,16,44],boundari:9,bounds_error:3,box:[6,9,17,25,30,33,35,36],boxsiz:[6,8,9,23,30,33,35,36],boxsizepars:6,breakflag:16,broadcast:10,built:16,bulh:12,bunchsiz:[32,37,38],bypass:16,cach:16,calcul:[3,6],call:[4,6,9,12,16],callabl:16,caller:16,calul:9,can:[3,4,8,12,16],captur:16,care:[9,16],caseless:16,caselesskeyword:16,caselessliter:16,catalogu:8,cell:46,center:[4,8,9,23],centerofmass:9,central:[30,35,36],charact:[16,43],charsnotin:16,chase:16,check:16,checkrecurs:16,choos:10,chunk:[5,6,8,17],classmethod:[4,6,7,17,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,43,44,46],clear:16,clearli:16,client:16,clip:16,close:16,closer:16,closest:4,cloud:46,cluster:8,clutter:16,code:16,col:[6,16,43,44],collect:[4,6,8,9],colno:16,column:[4,6,7,16,30,31,32,34,35,36,37,38,43],com:4,combin:[12,16],comm:[5,8,9,10,17],comm_world:10,command:[6,12,17,24],commandlin:35,comment:[12,16,35,36,43],common:16,commun:[5,8,9,10],compar:16,compat:16,compil:16,compiledretyp:16,complet:16,complex:[6,16,27,43,44],compos:16,comput:[4,8,10,17,24,25],compute_3d_corr:10,compute_3d_pow:10,compute_brutal_corr:10,concaten:16,config:[12,17],config_fil:6,configur:[10,46],conflict:16,consid:8,consist:[12,16],constant:16,construct:[4,16,25],constructor:[4,16],contain:[8,12,16],content:16,contigu:16,convent:3,convers:24,convert:[8,11,16,24],convert_args_file_to_arg:12,convolut:46,coord:4,coordin:4,copi:[3,4,16],corr1ddataset:4,corr2ddataset:4,corr:[4,24],correl:[3,4,10,24,25],correspond:[8,10,11,16,41,43,44],corrfrompow:[0,1,2],cosin:10,cosmolog:8,could:[8,16,30,35,36],count:[5,9,10,16,24],countedarrai:16,cpus_per_work:17,crash:16,creat:[7,12,16,17],current:[4,5,6,16],custom:16,damp:3,dash:16,data:[4,5,6,7,8,10,16,21,22,23,24,25,26,27,28,30,32,33,35,36,37,38,40,41,43,44],datafram:35,dataset:[0,1,2],datasourc:[0,6,8,10,18,19,25],davi:8,dbscan:8,debug:[10,16],debugg:16,declar:[6,16],decomposit:10,decor:[6,16],default_keyword_char:16,default_white_char:16,defaultpaint:[6,19,39],defaultvalu:16,defin:[6,8,9,10,16],definit:16,delim:16,delimit:16,delimitedlist:16,dens:11,dense_label:11,densiti:[35,36],deprec:16,depth:16,derricw:4,desc:17,describ:[6,21],descript:17,design:[4,6,12,33],desir:[4,10,16,17],determin:40,diagnost:16,dict:[4,12,16,24,44],dictionari:[4,16,17,24,25],dictof:16,differ:[6,16],dim:[4,6],dimens:[4,6,10,24,30,35,36,44],direct:[3,10,24,30,35,36],directli:16,disabl:16,disallow:16,displai:16,distort:[30,35,36],distribut:5,distributedarrai:[0,1,2],div:16,divid:[17,27,46],doaction:16,doc:16,doctag:16,document:[4,10],doe:[3,8,10,16,21,40,41],don:[16,41],done:[3,8,16,33],doubl:16,down:10,downcasetoken:16,drop:12,due:5,dump:16,dure:16,each:[4,5,6,8,9,10,11,16,17,24],easili:16,edg:[4,10,24,25,43,44],effect:[16,27,46],either:[6,10,30,35,36],elem:16,element:[4,16],embed:16,empti:[5,6,12,16],emptyrank:5,emptyranktyp:5,emul:16,enabl:16,enablepackrat:16,enclos:16,end:[10,16,44],endquotechar:16,ensur:16,entir:16,entri:4,equal:[10,30,35,36],equiv_class:11,equival:[11,16],err:16,error:16,errorstop:16,escap:16,escchar:16,escquot:16,especi:16,estim:3,etc:16,evalu:[16,17],even:16,everi:[6,10],exact:[4,16],exactli:16,exampl:[4,5,11,16,30,35,36],examplealgorithm:[19,20],except:[4,8,10,16],exceptionact:16,exclud:16,excludechar:16,exclus:10,execut:16,exist:[16,17],expand:16,expandtab:16,explanatori:16,explicit:16,explicitli:[4,16],explictli:16,expr:16,express:16,extend:[10,16],extens:[6,16,35],extensionpoint:[0,1,2,4],extra:[16,17],extract:16,factor:[4,35,36],fail:16,failon:16,fals:[3,4,5,6,8,10,11,16,31,32,34,37,38],fashion:4,faster:8,fastpm:[6,19,29],fastpmdatasourc:[6,31],featur:[8,16],fetch:5,fft:25,fftcorrel:[6,25],fftcorrelationalgorithm:[6,25],fftpower:[6,25],fftpoweralgorithm:[6,25],fid:7,field:[6,10,16,25,30,33,35,36,40,41,46],fieldstyp:25,file:[0,1,2,4,6],file_or_filenam:16,filenam:[4,12,16,24],fill_valu:3,filter:16,finalize_attribut:[21,30,31],finder:[6,8],first:[4,5,8,16,17,24,25],fitpack2:3,fix:4,flag:16,flatten:41,fmin:11,fof6d:[6,19,20],fof6dalgorithm:[6,23],fof:[0,1,2,6],fofalgorithm:[6,22],fofdatasourc:[6,30],fofgroup:[6,19,29],follow:[4,6,16],followedbi:16,forc:4,form:16,format:[8,12,16,17],forward:16,found:16,fourier:[3,6,10,46],friend:8,from:[3,4,5,6,8,10,11,12,16,30,33,35,36,41],from_1d_measur:4,from_2d_measur:4,from_nbkit:4,fromstr:6,ftype:35,full:[6,16,31,32,34,37,38],fwdexpr:16,gadget:[3,6,8,19,29],gadgetdatasourc:[6,32],gadgetgrouptab:[6,32],gadgetgrouptabdatasourc:[6,32],gaussian:3,gener:[6,16,17],get:16,getnam:16,gist:4,github:4,give:6,given:[12,16,17,41],global:[5,6,12],gotocolumn:16,grammar:16,greet:16,grid:[4,6,10,19,29],griddatasourc:[6,33],gridpaint:[6,19,33,39],group:[16,17,30,35],guess:35,h5py:30,had:11,half:10,halo:[0,1,2,6,8],haloid:23,halolabel:[6,19,29],handi:11,handl:16,happen:16,haskei:16,have:[3,4,8,9,16,17,41],hdf5:[30,35],head:5,hello:16,help:16,help_formatt:6,helper:[5,11,16],helpformatt:6,helpformattercolon:[6,12],here:[8,16],hex:16,hierach:8,hierarchi:16,high:6,hold:[4,24,25,43,44],how:[4,16,30,35,36],hpo:9,html:16,http:[4,16],ident:11,identchar:16,identifi:16,ifandonlyif:16,ignor:16,ignoreexpr:16,imag:[43,44],immedi:16,implement:[6,8],implicitli:16,improperli:16,inaccuraci:4,inbuilt:16,includ:[10,16],inclus:[10,16],inconsist:16,indent:16,indent_incr:12,indentedblock:16,indentstack:16,index:[4,16],indic:[4,16],indirectli:16,individu:12,inexact:4,inf:[4,11],infinit:16,infix:16,infixnot:16,info:41,inform:16,initchar:16,initi:[16,17,24],inlineliteralsus:16,input:[3,4,6,10,11,12,16,40],input_field:25,insensit:16,insert:16,inspect:16,insstr:16,instal:[30,35],instanc:[4,6,10,16,17],instanti:25,instead:[3,4,16],instr:16,integ:[3,4,5,8,9,10,16],integr:[3,4],intent:6,interfac:[6,16],intern:[3,11,16,35],interp1d:3,interpol:3,interpolatedunivariatesplin:3,interpret:17,interven:16,intexpr:16,intracomm:[5,8,9],invok:16,isinst:16,isotrop:[10,30,35,36],item:[5,10,16],itemseq:16,iter:[6,16,17],iteritem:16,iterkei:16,itervalu:16,itself:16,job:17,join:16,joinstr:16,just:[4,12,16],k_cen:4,kdcount:[8,10],keep:[4,16],keeporiginaltext:16,kei:[4,5,6,16,17,24],kernel:[3,46],keyword:[4,16,43],kind:3,klass:6,known:[6,8,16],kwarg:[4,6,12,16,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,46],kwd:6,label:[5,8,9,11],larg:[6,8,12],larger:4,last:[5,10,16],later:16,lead:16,learn:[3,16],least:16,leav:16,leavewhitespac:16,left:16,len:[11,16],length:[4,8,16],less:[4,8],letter:16,level:[6,10,16],lex:16,librari:16,lightweight:4,like:[4,16],limit:5,line:[6,10,12,16,17,24,35,36,43],lineartopolog:5,lineend:16,lineno:16,linestart:16,link:8,linking_length:8,list:[4,5,10,12,16,17,24,25,30,35,36,43,44],list_str:[35,36],listallmatch:16,listpluginsact:12,liter:16,literalstringclass:16,load:12,loadtxt:43,loc:16,local:[5,8],local_fof:8,locat:16,locatedexpr:16,locn_end:16,locn_start:16,log:[3,10,21,25],log_level:[8,10,17],logger:[21,25],logic:16,logscal:3,longest:16,look:16,lookahead:16,loop:6,lower:16,lpar:16,made:16,mai:16,main:16,maintain:16,make:16,makehtmltag:16,makexmltag:16,manag:[16,17],mani:16,manner:12,mark:16,marker:16,markerstr:16,markinputlin:16,martin:8,mask:4,mass:[8,9,30,35,36],massiv:8,master:17,match:[4,16],matchexpr:16,matchfirst:16,matchonlyatcol:16,matchpreviousexpr:16,matchpreviousliter:16,matchstr:16,max:16,max_help_posit:12,maximum:[10,16],maxmatch:16,mean:[4,8,9,16,24,27,46],measur:[4,25,43,44],measurement1dstorag:[6,19,42],measurement2dstorag:[6,19,42],measurementstorag:[4,6,43,44],measurestat:[0,1,2],member:[6,16],memoiz:16,memori:8,merg:8,mesh:[10,40,41],messag:16,meta:[4,6,7,43,44],metadata:[4,43,44],method:[4,5,6,16],methodcal:16,might:16,min:16,minid:8,minim:8,minimum:[8,16],minlength:11,miss:16,modal:16,mode:4,model:4,modifi:[12,16],modul:[0,1,2],moment:41,more:[16,30,35,36],most:[8,16],mount:6,mpi4pi:[5,8,9],mpi:[5,8,9,10,17],mpsort:5,msg:16,mu_cen:4,multilin:16,multipl:[4,16],multipli:[4,35,36],multipol:[10,24,25],must:[4,5,6,8,10,16,30,35,36],myend:7,mystart:7,n_rank:17,name:[4,6,12,16,17,24,30,35,36,43,44],nameditemsonli:16,namespac:[6,12,16],nan:[3,4],nbar:23,ndarrai:[0,1,2,4],nearest:4,necessari:[6,12],need:[16,17,24],neglect:[32,37,38],nest:16,nestedexpr:16,never:16,new_nam:4,new_shap:4,newarr:11,newlin:16,next:5,nmesh:41,nmin:8,nmu:10,nobject:11,nois:8,nomatch:16,non:16,none:[3,4,5,6,9,10,11,12,16,17,35,36],normal:[3,10,16],normalizedc:[6,27,46],notani:16,notat:16,notchar:16,note:[3,4,6,8,10,16,30,33,35,36,43],noth:[16,21],now:4,npart:7,ns1:16,ns2:16,ntot:[6,23,40,41],nulldebugact:16,num:16,number:[4,5,8,9,10,16,17,30,35,36,40,41],numer:[3,10],numpi:[4,5,11,36,43],numterm:16,object:[4,5,8,9,10,11,16,17,21,25,40,41],octal:16,often:16,old:4,old_nam:4,omit:16,onc:[6,16],oneof:16,oneormor:16,onli:[4,5,10,16,35,36],onlyonc:16,onto:[10,35,36,40,41],opassoc:16,open:[6,16],oper:[4,6,8,9,11,16],operatorpreced:16,opexpr:16,oplist:16,optim:16,option:[4,10,12,16,17,30,35,36],order:[4,8,16],orderbi:5,ordereddict:4,org:16,origin:[12,16],originaltextfor:16,other:[5,16,35,36],otherwis:[4,6,16,30,35,36],out:[10,11,16,43,44],output:[4,11,16,21,22,23,24,25,26,27,28,43],over:[4,6,16,17],overlap:[4,16],overlook:16,overrid:[6,16],overridden:16,p3d:10,packrat:16,paint:[6,10,35,36,40,41],painter:[0,6,10,18,19,25],pair:[4,10,24],paircount:10,paircountcorrel:[6,19,20],paircountcorrelationalgorithm:[6,24],panda:[6,19,29,30],pandasdatasourc:[6,35],parallel:[8,17,32,37,38],paramet:[4,5,8,9,10,11,12,16,17,24,25,30,33,35,36,40,41,43,44],parenthes:16,pars:[6,12,16,17],parse_arg:17,parse_known_yaml:6,parseabl:16,parseact:16,parseal:16,parsebaseexcept:16,parseelementenh:16,parseelementlist:16,parseexcept:16,parseexpress:16,parsefatalexcept:16,parsefil:16,parseimpl:16,parser:[6,12,16,17,24],parserel:16,parseresult:16,parsestr:16,parsesyntaxexcept:16,parsewithtab:16,part:[4,6],particl:[6,8,9,10,40,41],particlemesh:[10,33,40,41],pass:[3,4,10,12,16],path:[6,12,30,33,35,36,43,44],pattern:16,pdb:16,per:17,perform:[4,6,12,16],period:[9,25],periodicbox:[19,20],pkmu:4,place:[4,16],plain:[36,44],plaintext:[6,19,29],plaintextdatasourc:[6,36],plane:10,plugin:[0,6,12],plugin_nam:[6,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,43,44,46],plugininterfac:6,pluginmanag:[0,1,2],pluginmount:6,point:[3,4,6,10],pole:10,pop:16,poscol:[35,36],posf:[35,36],posit:[6,8,9,16,35,36],possibl:[4,17],post:16,postpars:16,power1ddataset:4,power2ddataset:4,power:[3,4,6,10,25],powerspectrum:3,pprint:16,preced:16,prefix:16,prepars:16,preserv:16,pretti:16,prev:5,previou:16,print:[11,16],printabl:16,printer:16,problem:16,procedur:4,process:16,prog:12,program:16,project:10,project_to_basi:10,proper:16,provid:[4,6,10,16,17,30,35,36],pstr:16,psyco:16,punctuat:16,pypars:16,python:16,qualifi:16,quantiti:[33,41,46],quickli:16,quit:16,quot:16,quotat:16,quotechar:16,quotedstr:16,radial:3,rais:[4,16],random:10,rang:[10,16,17],rank:[5,6,9,17],raw:16,rawtexthelpformatt:12,rbin:10,read1dplaintext:4,read2dplaintext:4,read:[4,6,7,16,17,30,31,32,33,34,35,36,37,38,41,43],read_and_decompos:6,read_csv:35,read_hdf5:35,readabl:16,readal:[6,30,35,36],real:[10,41,43,44],realli:11,reason:16,rebin:4,recarrai:36,recfromtxt:36,reclassifi:8,recogn:16,recommend:16,recurs:16,recursivegrammarexcept:16,redshift:[30,35,36],reduc:[6,10],refer:0,referenc:16,regard:9,regardless:16,regex:16,regexp:16,regist:[6,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,43,44,46],registri:6,regular:16,reindex:4,rel:16,remov:[4,8,16,17,27,46],removedc:[6,46],removequot:16,renam:4,rename_vari:4,repeat:16,repeatedli:16,repetit:16,replac:[3,11,16],replacehtmlent:16,replacements_from_fil:17,replacesort:11,replacewith:16,replstr:16,report:16,repres:[4,6,8,16,40,41],represent:[12,16,25],requir:[4,16],reserv:16,reset:16,resetcach:16,reshap:11,respect:[10,16],restor:16,restrict:16,result:[4,11,16,17,24,25],resultsnam:16,return_label:8,return_spac:4,revert:16,right:16,rightleftassoc:16,rmax:10,root:6,routin:[8,11],rpar:16,rsd:[30,35,36],rule:3,run:[4,6,8,12,17,21,22,23,24,25,26,27,28],run_al:17,runtim:16,salut:16,same:[5,6,8,9,10,16],sampl:10,save:[6,21,22,23,24,25,26,27,28],savelist:16,scalar:[4,5],scan:16,scanstr:16,scienc:8,scipi:3,scrape:16,script:12,search:16,searchstr:16,second:[4,12,16,24,25],section:6,see:[4,10,16],sel:4,select:[4,30,35,36],selectionlanguag:[2,15],self:[4,5,6,16],semant:16,sensit:16,separ:[4,6,10,16,35,36],seper:[8,12],sequenc:[8,16],sequenti:[8,17],seri:[12,16],set:[4,6,10,12,16,17,41,46],setbreak:16,setdebug:16,setdebugact:16,setdefaultkeywordchar:16,setdefaultwhitespacechar:16,setfailact:16,setnam:16,setparseact:16,setresultsnam:16,setup:24,setwhitespacechar:16,shall:6,shape:[4,10,30,35,36],share:16,should:[6,10,16],show:4,side:16,sight:10,sigma_8:3,signatur:16,signific:16,simpl:[5,6,16],simpler:16,simpli:[12,16,17],simplifi:16,simul:8,sinc:[4,8,16,41],singl:[6,16,30,35,36],size:[4,6,8,9,30,33,35,36],skip:[3,16,43],skipto:16,slice:4,small:6,smaller:4,smooth:27,some:16,sort:[5,8,11],sourc:[3,4,5,6,7,8,9,10,11,12,16,17,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,43,44,46],space:[4,6,10,12,16,30,35,36,46],span:5,spc:16,special:16,specif:[4,16],specifi:[4,10,12,16,24,25,30,35,36,40,41,43,44],spectrum:[3,4,6,10,25],speedup:16,split:[8,17],split_rank:17,split_size_2d:8,springel:8,squeez:4,srang:16,sre_pattern:16,stack:[4,16],stage:12,standand:8,start:[5,8,11,16,43],startact:16,startloc:16,stat:[6,31,32,34,37,38],statement:16,statementwithindentedblock:16,statist:[6,10],step:12,stop:16,storag:[0,18,19],store:[4,35,36],str:[4,6,10,16,17,24,25,30,33,35,36],straight:33,streamlin:16,strg:16,string:[4,6,12,16,17,24,25,30,35,36],stringend:16,stringstart:16,stripe:5,stripedfil:[0,1,2,7],stripefil:7,structur:[4,8,16],subclass:[6,16],subfof:23,submodul:[0,1],subpackag:[0,1],subpars:6,subroutin:8,subsampl:[6,10,19,20],subsequ:6,subset:[30,35,36],substr:16,successact:16,successfulli:16,suffix:4,suggest:16,sum:4,summat:10,support:[8,16],suppress:16,sure:16,symbol:16,symmetr:10,syntax:[4,12,16],tab:16,tabular:16,tag:16,tagstr:16,tail:5,take:[4,6,16],taken:9,target:[4,16],task:17,task_dim:17,task_valu:17,taskmanag:[2,15],tasks_pars:17,tell:16,term:[8,16],terribl:12,test:[5,16],text:[16,30,35,36,44],than:[4,8,16],thei:[5,8,16],them:[11,16,43],therebi:16,thi:[3,4,5,6,8,9,10,12,16,17,24,25,27,30,35,36,43],thing:11,those:[3,4,6,16],three:[30,35,36],thresh:[8,23],through:[3,4],throwabl:16,thrown:16,tidaltensor:[6,19,20],time:[3,4],tok:16,token:16,tokenconvert:16,tokenlist:16,toklist:16,too:8,topolog:5,total:[10,12,17,40,41],tpmlabel:[6,19,29],tpmsnapshot:[6,19,29],tpmsnapshotdatasourc:[6,38],tpmsnapshotfil:7,tracehalo:[6,19,20],tracehaloalgorithm:[6,28],traceparseact:16,tradit:16,transfer:[0,6,10,18,19,25],transferfunct:[19,45],transform:[3,10,16],transformstr:16,transfrom:6,trapz:3,treat:[10,16,35,36,43,44],trn:16,trypars:16,tupl:[4,10,16,24,25],two:[8,10,12,16,43,44],type:[5,8,12,24,30,35,36],typic:[8,16],ufunc:11,unari:16,unbacktrack:16,uncollect:6,undefin:[16,30,35,36],underli:8,undo:16,ungroup:16,uniqu:[5,11],unique_label:5,unit:[4,6,10],univariatesplin:3,unknown:6,unquoteresult:16,until:16,untoken:16,upcas:16,upcasetoken:16,updat:6,upper:16,usag:12,usecol:[35,36],user:16,useregex:16,usual:3,util:[0,1,2],valid:[4,16],validatetrac:16,valu:[4,5,6,8,10,11,16,17,24,35,36,41,43],variabl:[4,16],variou:6,vel:23,velcol:[35,36],velf:[35,36],veloc:[6,8,35,36],velocii:[35,36],verbose_stacktrac:16,verifi:16,version:8,vertic:4,vex:16,vfactor:23,via:[4,12,24,35],view:16,volker:8,want:16,wavenumb:10,weight:4,well:[4,5],were:[8,16],when:[3,4,10,12,16],where:[4,6,10,16],whether:16,which:[3,4,6,8,10,12,16,21,25,33,41,43,46],white:[8,16],whitespac:16,whitestr:16,width:12,window:46,withattribut:16,within:16,without:16,word:16,wordchar:16,wordend:16,wordstart:16,work:16,worker:17,world:16,would:16,wrapper:16,write:[6,7,16,43,44],write_hdf5:[26,27],write_mwhite_subsampl:26,written:43,x0c:16,x21:16,x3d:10,xiaoi:3,xml:16,xrai:4,y3d:10,yacc:16,yaml:[6,12],yield:6,you:[16,30,35,36],your:16,zero:[16,46],zeroormor:16},titles:["nbodykit package","API Reference","nbodykit package","nbodykit.corrfrompower module","nbodykit.dataset module","nbodykit.distributedarray module","nbodykit.extensionpoints module","nbodykit.files module","nbodykit.fof module","nbodykit.halos module","nbodykit.measurestats module","nbodykit.ndarray module","nbodykit.pluginmanager module","nbodykit.plugins module","nbodykit.stripedfile module","nbodykit.utils package","nbodykit.utils.selectionlanguage module","nbodykit.utils.taskmanager module","Plugins Reference","plugins package","plugins.algorithms package","plugins.algorithms.ExampleAlgorithm module","plugins.algorithms.FOF module","plugins.algorithms.FOF6D module","plugins.algorithms.PairCountCorrelation module","plugins.algorithms.PeriodicBox module","plugins.algorithms.Subsample module","plugins.algorithms.TidalTensor module","plugins.algorithms.TraceHalo module","plugins.datasource package","plugins.datasource.FOFGroups module","plugins.datasource.FastPM module","plugins.datasource.Gadget module","plugins.datasource.Grid module","plugins.datasource.HaloLabel module","plugins.datasource.Pandas module","plugins.datasource.PlainText module","plugins.datasource.TPMLabel module","plugins.datasource.TPMSnapshot module","plugins.painter package","plugins.painter.DefaultPainter module","plugins.painter.GridPainter module","plugins.storage package","plugins.storage.Measurement1DStorage module","plugins.storage.Measurement2DStorage module","plugins.transfer package","plugins.transfer.TransferFunction module"],titleterms:{algorithm:[20,21,22,23,24,25,26,27,28],api:1,corrfrompow:3,dataset:4,datasourc:[29,30,31,32,33,34,35,36,37,38],defaultpaint:40,distributedarrai:5,examplealgorithm:21,extensionpoint:6,fastpm:31,file:7,fof6d:23,fof:[8,22],fofgroup:30,gadget:32,grid:33,gridpaint:41,halo:9,halolabel:34,measurement1dstorag:43,measurement2dstorag:44,measurestat:10,modul:[3,4,5,6,7,8,9,10,11,12,13,14,16,17,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,40,41,43,44,46],nbodykit:[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],ndarrai:11,packag:[0,2,15,19,20,29,39,42,45],painter:[39,40,41],paircountcorrel:24,panda:35,periodicbox:25,plaintext:36,plugin:[13,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46],pluginmanag:12,refer:[1,18],selectionlanguag:16,storag:[42,43,44],stripedfil:14,submodul:[2,15,20,29,39,42,45],subpackag:[2,19],subsampl:26,taskmanag:17,tidaltensor:27,tpmlabel:37,tpmsnapshot:38,tracehalo:28,transfer:[45,46],transferfunct:46,util:[15,16,17]}})