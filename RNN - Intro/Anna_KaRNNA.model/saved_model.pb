±§-
¤ô
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ü+

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	S*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	S*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	S*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	S*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:S*
dtype0
ª
&rnn/stacked_rnn_cells/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&rnn/stacked_rnn_cells/lstm_cell/kernel
£
:rnn/stacked_rnn_cells/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp&rnn/stacked_rnn_cells/lstm_cell/kernel* 
_output_shapes
:
*
dtype0
¾
0rnn/stacked_rnn_cells/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20rnn/stacked_rnn_cells/lstm_cell/recurrent_kernel
·
Drnn/stacked_rnn_cells/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp0rnn/stacked_rnn_cells/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0
¡
$rnn/stacked_rnn_cells/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$rnn/stacked_rnn_cells/lstm_cell/bias

8rnn/stacked_rnn_cells/lstm_cell/bias/Read/ReadVariableOpReadVariableOp$rnn/stacked_rnn_cells/lstm_cell/bias*
_output_shapes	
:*
dtype0
®
(rnn/stacked_rnn_cells/lstm_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(rnn/stacked_rnn_cells/lstm_cell_1/kernel
§
<rnn/stacked_rnn_cells/lstm_cell_1/kernel/Read/ReadVariableOpReadVariableOp(rnn/stacked_rnn_cells/lstm_cell_1/kernel* 
_output_shapes
:
*
dtype0
Â
2rnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42rnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel
»
Frnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp2rnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel* 
_output_shapes
:
*
dtype0
¥
&rnn/stacked_rnn_cells/lstm_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&rnn/stacked_rnn_cells/lstm_cell_1/bias

:rnn/stacked_rnn_cells/lstm_cell_1/bias/Read/ReadVariableOpReadVariableOp&rnn/stacked_rnn_cells/lstm_cell_1/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
Ð
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueBþ B÷
ä
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
 
b


embeddings
trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
 
?

0
1
2
3
4
5
 6
7
8
?

0
1
2
3
4
5
 6
7
8
­
!layer_metrics
"non_trainable_variables
#layer_regularization_losses

$layers
regularization_losses
	variables
%metrics
trainable_variables
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE


0
 


0
­
trainable_variables
&layer_metrics
'non_trainable_variables

(layers
regularization_losses
	variables
)metrics
*layer_regularization_losses
]
	+cells
,trainable_variables
-regularization_losses
.	variables
/	keras_api
 
 
*
0
1
2
3
4
 5
*
0
1
2
3
4
 5
¹
0layer_metrics
1non_trainable_variables

2states
3layer_regularization_losses

4layers
regularization_losses
	variables
5metrics
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
trainable_variables
6layer_metrics
7non_trainable_variables

8layers
regularization_losses
	variables
9metrics
:layer_regularization_losses
b`
VARIABLE_VALUE&rnn/stacked_rnn_cells/lstm_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE0rnn/stacked_rnn_cells/lstm_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$rnn/stacked_rnn_cells/lstm_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(rnn/stacked_rnn_cells/lstm_cell_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE2rnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&rnn/stacked_rnn_cells/lstm_cell_1/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
3
 
 
 
 
 
 

;0
<1
*
0
1
2
3
4
 5
 
*
0
1
2
3
4
 5
­
,trainable_variables
=layer_metrics
>non_trainable_variables

?layers
-regularization_losses
.	variables
@metrics
Alayer_regularization_losses
 
 

B0
C1
 

0
 
 
 
 
 
 

D
state_size

kernel
recurrent_kernel
bias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api

I
state_size

kernel
recurrent_kernel
 bias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
 
 

;0
<1
 
 
 
 
 

0
1
2
 

0
1
2
­
Etrainable_variables
Nlayer_metrics
Onon_trainable_variables

Players
Fregularization_losses
G	variables
Qmetrics
Rlayer_regularization_losses
 

0
1
 2
 

0
1
 2
­
Jtrainable_variables
Slayer_metrics
Tnon_trainable_variables

Ulayers
Kregularization_losses
L	variables
Vmetrics
Wlayer_regularization_losses
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1embedding/embeddings&rnn/stacked_rnn_cells/lstm_cell/kernel0rnn/stacked_rnn_cells/lstm_cell/recurrent_kernel$rnn/stacked_rnn_cells/lstm_cell/bias(rnn/stacked_rnn_cells/lstm_cell_1/kernel2rnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel&rnn/stacked_rnn_cells/lstm_cell_1/biasdense/kernel
dense/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_61367284
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp:rnn/stacked_rnn_cells/lstm_cell/kernel/Read/ReadVariableOpDrnn/stacked_rnn_cells/lstm_cell/recurrent_kernel/Read/ReadVariableOp8rnn/stacked_rnn_cells/lstm_cell/bias/Read/ReadVariableOp<rnn/stacked_rnn_cells/lstm_cell_1/kernel/Read/ReadVariableOpFrnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel/Read/ReadVariableOp:rnn/stacked_rnn_cells/lstm_cell_1/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_61369674
Ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/bias&rnn/stacked_rnn_cells/lstm_cell/kernel0rnn/stacked_rnn_cells/lstm_cell/recurrent_kernel$rnn/stacked_rnn_cells/lstm_cell/bias(rnn/stacked_rnn_cells/lstm_cell_1/kernel2rnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel&rnn/stacked_rnn_cells/lstm_cell_1/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_61369711´»*


4__inference_stacked_rnn_cells_layer_call_fn_61369395

inputs

states_0_0

states_0_1

states_1_0

states_1_1
unknown:

	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
identity

identity_1

identity_2

identity_3

identity_4¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputs
states_0_0
states_0_1
states_1_0
states_1_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout	
2*
_collective_manager_ids
 *x
_output_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_613654882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/1
«

,__inference_embedding_layer_call_fn_61368017

inputs
unknown:	S
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_613663722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
í
B__inference_test_layer_call_and_return_conditional_losses_61367165

inputs%
embedding_61367143:	S 
rnn_61367146:
 
rnn_61367148:

rnn_61367150:	 
rnn_61367152:
 
rnn_61367154:

rnn_61367156:	!
dense_61367159:	S
dense_61367161:S
identity¢dense/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢rnn/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_61367143*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_613663722#
!embedding/StatefulPartitionedCallö
rnn/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0rnn_61367146rnn_61367148rnn_61367150rnn_61367152rnn_61367154rnn_61367156*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613670922
rnn/StatefulPartitionedCall¹
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_61367159dense_61367161*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_613666632
dense/StatefulPartitionedCallé
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


while_body_61368518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
X
Dwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
R
Cwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
Z
Fwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
T
Ewhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorT
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
P
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	V
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
R
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
1while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/ones_like/Shape«
1while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1while/stacked_rnn_cells/lstm_cell/ones_like/Const
+while/stacked_rnn_cells/lstm_cell/ones_likeFill:while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0:while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/ones_like§
/while/stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?21
/while/stacked_rnn_cells/lstm_cell/dropout/Const
-while/stacked_rnn_cells/lstm_cell/dropout/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:08while/stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell/dropout/MulÆ
/while/stacked_rnn_cells/lstm_cell/dropout/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:21
/while/stacked_rnn_cells/lstm_cell/dropout/Shapeº
Fwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform8while/stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ì¼2H
Fwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform¹
8while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2:
8while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yÇ
6while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualOwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0Awhile/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualæ
.while/stacked_rnn_cells/lstm_cell/dropout/CastCast:while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.while/stacked_rnn_cells/lstm_cell/dropout/Cast
/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul1while/stacked_rnn_cells/lstm_cell/dropout/Mul:z:02while/stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_1/Const
/while/stacked_rnn_cells/lstm_cell/dropout_1/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_1/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_1/Shape¿
Hwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÿê2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_1/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_1/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_2/Const
/while/stacked_rnn_cells/lstm_cell/dropout_2/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_2/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_2/Shape¿
Hwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÍôJ2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_2/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_2/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_3/Const
/while/stacked_rnn_cells/lstm_cell/dropout_3/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_3/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_3/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed22J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_3/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_3/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1ï
%while/stacked_rnn_cells/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/mul÷
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpý
(while/stacked_rnn_cells/lstm_cell/MatMulMatMul)while/stacked_rnn_cells/lstm_cell/mul:z:0?while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/MatMulý
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpí
*while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulwhile_placeholder_2Awhile/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell/MatMul_1ô
%while/stacked_rnn_cells/lstm_cell/addAddV22while/stacked_rnn_cells/lstm_cell/MatMul:product:04while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/addõ
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02:
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
)while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd)while/stacked_rnn_cells/lstm_cell/add:z:0@while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/BiasAdd¨
1while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1while/stacked_rnn_cells/lstm_cell/split/split_dimË
'while/stacked_rnn_cells/lstm_cell/splitSplit:while/stacked_rnn_cells/lstm_cell/split/split_dim:output:02while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'while/stacked_rnn_cells/lstm_cell/splitÆ
)while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid0while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/SigmoidÊ
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Ò
'while/stacked_rnn_cells/lstm_cell/mul_1Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_1½
&while/stacked_rnn_cells/lstm_cell/TanhTanh0while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/stacked_rnn_cells/lstm_cell/Tanhç
'while/stacked_rnn_cells/lstm_cell/mul_2Mul-while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0*while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_2è
'while/stacked_rnn_cells/lstm_cell/add_1AddV2+while/stacked_rnn_cells/lstm_cell/mul_1:z:0+while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/add_1Ê
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2¼
(while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh+while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/Tanh_1ë
'while/stacked_rnn_cells/lstm_cell/mul_3Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0,while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_3Å
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape+while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape¯
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Const
-while/stacked_rnn_cells/lstm_cell_1/ones_likeFill<while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0<while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/ones_like«
1while/stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell_1/dropout/Const
/while/stacked_rnn_cells/lstm_cell_1/dropout/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0:while/stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell_1/dropout/MulÌ
1while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2­å2J
Hwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell_1/dropout/CastCast<while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell_1/dropout/Cast
1while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul3while/stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:04while/stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeÆ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¨ 2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_1/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeÆ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2è2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_2/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeÆ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Þõ³2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_3/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1ð
'while/stacked_rnn_cells/lstm_cell_1/mulMul+while/stacked_rnn_cells/lstm_cell/mul_3:z:05while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/mulý
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
*while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul+while/stacked_rnn_cells/lstm_cell_1/mul:z:0Awhile/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/MatMul
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpó
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulwhile_placeholder_4Cwhile/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1ü
'while/stacked_rnn_cells/lstm_cell_1/addAddV24while/stacked_rnn_cells/lstm_cell_1/MatMul:product:06while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/addû
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02<
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
+while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd+while/stacked_rnn_cells/lstm_cell_1/add:z:0Bwhile/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/BiasAdd¬
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimÓ
)while/stacked_rnn_cells/lstm_cell_1/splitSplit<while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:04while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)while/stacked_rnn_cells/lstm_cell_1/splitÌ
+while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/SigmoidÐ
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ø
)while/stacked_rnn_cells/lstm_cell_1/mul_1Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_1Ã
(while/stacked_rnn_cells/lstm_cell_1/TanhTanh2while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell_1/Tanhï
)while/stacked_rnn_cells/lstm_cell_1/mul_2Mul/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0,while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_2ð
)while/stacked_rnn_cells/lstm_cell_1/add_1AddV2-while/stacked_rnn_cells/lstm_cell_1/mul_1:z:0-while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/add_1Ð
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Â
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh-while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1ó
)while/stacked_rnn_cells/lstm_cell_1/mul_3Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0.while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_3ñ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations9^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identity+while/stacked_rnn_cells/lstm_cell/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ø
while/Identity_5Identity+while/stacked_rnn_cells/lstm_cell/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ú
while/Identity_6Identity-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6ú
while/Identity_7Identity-while/stacked_rnn_cells/lstm_cell_1/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2r
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2x
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2z
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
Ù¦

A__inference_rnn_layer_call_and_return_conditional_losses_61368955

inputsN
:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
P
<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
J
;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	P
<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
R
>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
L
=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	
identity¢2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
+stacked_rnn_cells/lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/ones_like/Shape
+stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+stacked_rnn_cells/lstm_cell/ones_like/Constõ
%stacked_rnn_cells/lstm_cell/ones_likeFill4stacked_rnn_cells/lstm_cell/ones_like/Shape:output:04stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/ones_likeÆ
stacked_rnn_cells/lstm_cell/mulMulstrided_slice_2:output:0.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/mulã
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpå
"stacked_rnn_cells/lstm_cell/MatMulMatMul#stacked_rnn_cells/lstm_cell/mul:z:09stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/MatMulé
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpÖ
$stacked_rnn_cells/lstm_cell/MatMul_1MatMulzeros:output:0;stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell/MatMul_1Ü
stacked_rnn_cells/lstm_cell/addAddV2,stacked_rnn_cells/lstm_cell/MatMul:product:0.stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/addá
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpé
#stacked_rnn_cells/lstm_cell/BiasAddBiasAdd#stacked_rnn_cells/lstm_cell/add:z:0:stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/BiasAdd
+stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+stacked_rnn_cells/lstm_cell/split/split_dim³
!stacked_rnn_cells/lstm_cell/splitSplit4stacked_rnn_cells/lstm_cell/split/split_dim:output:0,stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!stacked_rnn_cells/lstm_cell/split´
#stacked_rnn_cells/lstm_cell/SigmoidSigmoid*stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/Sigmoid¸
%stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid*stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_1½
!stacked_rnn_cells/lstm_cell/mul_1Mul)stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_1«
 stacked_rnn_cells/lstm_cell/TanhTanh*stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 stacked_rnn_cells/lstm_cell/TanhÏ
!stacked_rnn_cells/lstm_cell/mul_2Mul'stacked_rnn_cells/lstm_cell/Sigmoid:y:0$stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_2Ð
!stacked_rnn_cells/lstm_cell/add_1AddV2%stacked_rnn_cells/lstm_cell/mul_1:z:0%stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/add_1¸
%stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid*stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_2ª
"stacked_rnn_cells/lstm_cell/Tanh_1Tanh%stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/Tanh_1Ó
!stacked_rnn_cells/lstm_cell/mul_3Mul)stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0&stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_3³
-stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape%stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Shape£
-stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Constý
'stacked_rnn_cells/lstm_cell_1/ones_likeFill6stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:06stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/ones_likeÙ
!stacked_rnn_cells/lstm_cell_1/mulMul%stacked_rnn_cells/lstm_cell/mul_3:z:00stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/mulé
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpí
$stacked_rnn_cells/lstm_cell_1/MatMulMatMul%stacked_rnn_cells/lstm_cell_1/mul:z:0;stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/MatMulï
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype027
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÞ
&stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulzeros_2:output:0=stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stacked_rnn_cells/lstm_cell_1/MatMul_1ä
!stacked_rnn_cells/lstm_cell_1/addAddV2.stacked_rnn_cells/lstm_cell_1/MatMul:product:00stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/addç
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpñ
%stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd%stacked_rnn_cells/lstm_cell_1/add:z:0<stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/BiasAdd 
-stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-stacked_rnn_cells/lstm_cell_1/split/split_dim»
#stacked_rnn_cells/lstm_cell_1/splitSplit6stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0.stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#stacked_rnn_cells/lstm_cell_1/splitº
%stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid,stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/Sigmoid¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ã
#stacked_rnn_cells/lstm_cell_1/mul_1Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_1±
"stacked_rnn_cells/lstm_cell_1/TanhTanh,stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell_1/Tanh×
#stacked_rnn_cells/lstm_cell_1/mul_2Mul)stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0&stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_2Ø
#stacked_rnn_cells/lstm_cell_1/add_1AddV2'stacked_rnn_cells/lstm_cell_1/mul_1:z:0'stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/add_1¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2°
$stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh'stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/Tanh_1Û
#stacked_rnn_cells/lstm_cell_1/mul_3Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0(stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61368826*
condR
while_cond_61368825*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1½
IdentityIdentitytranspose_1:y:03^stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2^stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp4^stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp5^stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4^stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp6^stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2h
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2f
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2l
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2n
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


4__inference_stacked_rnn_cells_layer_call_fn_61369424

inputs

states_0_0

states_0_1

states_1_0

states_1_1
unknown:

	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
identity

identity_1

identity_2

identity_3

identity_4¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinputs
states_0_0
states_0_1
states_1_0
states_1_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout	
2*
_collective_manager_ids
 *x
_output_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_613657832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/1
Ò
§
while_cond_61366489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61366489___redundant_placeholder06
2while_while_cond_61366489___redundant_placeholder16
2while_while_cond_61366489___redundant_placeholder26
2while_while_cond_61366489___redundant_placeholder36
2while_while_cond_61366489___redundant_placeholder46
2while_while_cond_61366489___redundant_placeholder56
2while_while_cond_61366489___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
éW
ª
A__inference_rnn_layer_call_and_return_conditional_losses_61365991

inputs.
stacked_rnn_cells_61365880:
.
stacked_rnn_cells_61365882:
)
stacked_rnn_cells_61365884:	.
stacked_rnn_cells_61365886:
.
stacked_rnn_cells_61365888:
)
stacked_rnn_cells_61365890:	
identity¢)stacked_rnn_cells/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2ó
)stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0stacked_rnn_cells_61365880stacked_rnn_cells_61365882stacked_rnn_cells_61365884stacked_rnn_cells_61365886stacked_rnn_cells_61365888stacked_rnn_cells_61365890*
Tin
2*
Tout	
2*
_collective_manager_ids
 *x
_output_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_613657832+
)stacked_rnn_cells/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_61365880stacked_rnn_cells_61365882stacked_rnn_cells_61365884stacked_rnn_cells_61365886stacked_rnn_cells_61365888stacked_rnn_cells_61365890*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61365901*
condR
while_cond_61365900*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1¥
IdentityIdentitytranspose_1:y:0*^stacked_rnn_cells/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)stacked_rnn_cells/StatefulPartitionedCall)stacked_rnn_cells/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
£
test_rnn_while_cond_61365205.
*test_rnn_while_test_rnn_while_loop_counter4
0test_rnn_while_test_rnn_while_maximum_iterations
test_rnn_while_placeholder 
test_rnn_while_placeholder_1 
test_rnn_while_placeholder_2 
test_rnn_while_placeholder_3 
test_rnn_while_placeholder_4 
test_rnn_while_placeholder_50
,test_rnn_while_less_test_rnn_strided_slice_1H
Dtest_rnn_while_test_rnn_while_cond_61365205___redundant_placeholder0H
Dtest_rnn_while_test_rnn_while_cond_61365205___redundant_placeholder1H
Dtest_rnn_while_test_rnn_while_cond_61365205___redundant_placeholder2H
Dtest_rnn_while_test_rnn_while_cond_61365205___redundant_placeholder3H
Dtest_rnn_while_test_rnn_while_cond_61365205___redundant_placeholder4H
Dtest_rnn_while_test_rnn_while_cond_61365205___redundant_placeholder5H
Dtest_rnn_while_test_rnn_while_cond_61365205___redundant_placeholder6
test_rnn_while_identity

test/rnn/while/LessLesstest_rnn_while_placeholder,test_rnn_while_less_test_rnn_strided_slice_1*
T0*
_output_shapes
: 2
test/rnn/while/Lessx
test/rnn/while/IdentityIdentitytest/rnn/while/Less:z:0*
T0
*
_output_shapes
: 2
test/rnn/while/Identity";
test_rnn_while_identity test/rnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
Ò
§
while_cond_61369133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61369133___redundant_placeholder06
2while_while_cond_61369133___redundant_placeholder16
2while_while_cond_61369133___redundant_placeholder26
2while_while_cond_61369133___redundant_placeholder36
2while_while_cond_61369133___redundant_placeholder46
2while_while_cond_61369133___redundant_placeholder56
2while_while_cond_61369133___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
	

&__inference_rnn_layer_call_fn_61368078

inputs
unknown:

	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613666192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

&__inference_rnn_layer_call_fn_61368061
inputs_0
unknown:

	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613659912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
É 
û
C__inference_dense_layer_call_and_return_conditional_losses_61369366

inputs4
!tensordot_readvariableop_resource:	S-
biasadd_readvariableop_resource:S
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	S*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:S2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2	
BiasAdd¥
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
§
while_cond_61365509
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61365509___redundant_placeholder06
2while_while_cond_61365509___redundant_placeholder16
2while_while_cond_61365509___redundant_placeholder26
2while_while_cond_61365509___redundant_placeholder36
2while_while_cond_61365509___redundant_placeholder46
2while_while_cond_61365509___redundant_placeholder56
2while_while_cond_61365509___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
éW
ª
A__inference_rnn_layer_call_and_return_conditional_losses_61365600

inputs.
stacked_rnn_cells_61365489:
.
stacked_rnn_cells_61365491:
)
stacked_rnn_cells_61365493:	.
stacked_rnn_cells_61365495:
.
stacked_rnn_cells_61365497:
)
stacked_rnn_cells_61365499:	
identity¢)stacked_rnn_cells/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2ó
)stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0stacked_rnn_cells_61365489stacked_rnn_cells_61365491stacked_rnn_cells_61365493stacked_rnn_cells_61365495stacked_rnn_cells_61365497stacked_rnn_cells_61365499*
Tin
2*
Tout	
2*
_collective_manager_ids
 *x
_output_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_613654882+
)stacked_rnn_cells/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0stacked_rnn_cells_61365489stacked_rnn_cells_61365491stacked_rnn_cells_61365493stacked_rnn_cells_61365495stacked_rnn_cells_61365497stacked_rnn_cells_61365499*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61365510*
condR
while_cond_61365509*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1¥
IdentityIdentitytranspose_1:y:0*^stacked_rnn_cells/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)stacked_rnn_cells/StatefulPartitionedCall)stacked_rnn_cells/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
§
while_cond_61368209
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61368209___redundant_placeholder06
2while_while_cond_61368209___redundant_placeholder16
2while_while_cond_61368209___redundant_placeholder26
2while_while_cond_61368209___redundant_placeholder36
2while_while_cond_61368209___redundant_placeholder46
2while_while_cond_61368209___redundant_placeholder56
2while_while_cond_61368209___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
þ	
ã
'__inference_test_layer_call_fn_61367307

inputs
unknown:	S
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:	S
	unknown_7:S
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_test_layer_call_and_return_conditional_losses_613666702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼Ù
É	
B__inference_test_layer_call_and_return_conditional_losses_61368010

inputs6
#embedding_embedding_lookup_61367610:	SR
>rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
T
@rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
N
?rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	T
@rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
V
Brnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
P
Arnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	:
'dense_tensordot_readvariableop_resource:	S3
%dense_biasadd_readvariableop_resource:S
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢embedding/embedding_lookup¢6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢	rnn/whilez
embedding/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding/Cast½
embedding/embedding_lookupResourceGather#embedding_embedding_lookup_61367610embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding/embedding_lookup/61367610*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding/embedding_lookup¡
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding/embedding_lookup/61367610*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#embedding/embedding_lookup/IdentityÈ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2'
%embedding/embedding_lookup/Identity_1t
	rnn/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2ú
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slicee
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessk
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	rnn/zerosi
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_1/mul/y
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lesso
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_1/packed/1
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/zeros_1i
rnn/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_2/mul/y
rnn/zeros_2/mulMulrnn/strided_slice:output:0rnn/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_2/mulk
rnn/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_2/Less/y
rnn/zeros_2/LessLessrnn/zeros_2/mul:z:0rnn/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_2/Lesso
rnn/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_2/packed/1
rnn/zeros_2/packedPackrnn/strided_slice:output:0rnn/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_2/packedk
rnn/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_2/Const
rnn/zeros_2Fillrnn/zeros_2/packed:output:0rnn/zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/zeros_2i
rnn/zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_3/mul/y
rnn/zeros_3/mulMulrnn/strided_slice:output:0rnn/zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_3/mulk
rnn/zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_3/Less/y
rnn/zeros_3/LessLessrnn/zeros_3/mul:z:0rnn/zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_3/Lesso
rnn/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_3/packed/1
rnn/zeros_3/packedPackrnn/strided_slice:output:0rnn/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_3/packedk
rnn/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_3/Const
rnn/zeros_3Fillrnn/zeros_3/packed:output:0rnn/zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/zeros_3}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm¸
rnn/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0rnn/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
rnn/TensorArrayV2/element_shapeÂ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ç
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
rnn/strided_slice_2®
/rnn/stacked_rnn_cells/lstm_cell/ones_like/ShapeShapernn/strided_slice_2:output:0*
T0*
_output_shapes
:21
/rnn/stacked_rnn_cells/lstm_cell/ones_like/Shape§
/rnn/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?21
/rnn/stacked_rnn_cells/lstm_cell/ones_like/Const
)rnn/stacked_rnn_cells/lstm_cell/ones_likeFill8rnn/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:08rnn/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell/ones_like£
-rnn/stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-rnn/stacked_rnn_cells/lstm_cell/dropout/Const
+rnn/stacked_rnn_cells/lstm_cell/dropout/MulMul2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:06rnn/stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/stacked_rnn_cells/lstm_cell/dropout/MulÀ
-rnn/stacked_rnn_cells/lstm_cell/dropout/ShapeShape2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2/
-rnn/stacked_rnn_cells/lstm_cell/dropout/Shape´
Drnn/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform6rnn/stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÔÒØ2F
Drnn/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformµ
6rnn/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6rnn/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y¿
4rnn/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualMrnn/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0?rnn/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4rnn/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualà
,rnn/stacked_rnn_cells/lstm_cell/dropout/CastCast8rnn/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,rnn/stacked_rnn_cells/lstm_cell/dropout/Castû
-rnn/stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul/rnn/stacked_rnn_cells/lstm_cell/dropout/Mul:z:00rnn/stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/stacked_rnn_cells/lstm_cell/dropout/Mul_1§
/rnn/stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?21
/rnn/stacked_rnn_cells/lstm_cell/dropout_1/Const
-rnn/stacked_rnn_cells/lstm_cell/dropout_1/MulMul2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:08rnn/stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/stacked_rnn_cells/lstm_cell/dropout_1/MulÄ
/rnn/stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:21
/rnn/stacked_rnn_cells/lstm_cell/dropout_1/Shapeº
Frnn/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform8rnn/stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2­´½2H
Frnn/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform¹
8rnn/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2:
8rnn/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yÇ
6rnn/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualOrnn/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0Arnn/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6rnn/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualæ
.rnn/stacked_rnn_cells/lstm_cell/dropout_1/CastCast:rnn/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/stacked_rnn_cells/lstm_cell/dropout_1/Cast
/rnn/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul1rnn/stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:02rnn/stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1§
/rnn/stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?21
/rnn/stacked_rnn_cells/lstm_cell/dropout_2/Const
-rnn/stacked_rnn_cells/lstm_cell/dropout_2/MulMul2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:08rnn/stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/stacked_rnn_cells/lstm_cell/dropout_2/MulÄ
/rnn/stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:21
/rnn/stacked_rnn_cells/lstm_cell/dropout_2/Shapeº
Frnn/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform8rnn/stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2±°á2H
Frnn/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform¹
8rnn/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2:
8rnn/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yÇ
6rnn/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualOrnn/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0Arnn/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6rnn/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualæ
.rnn/stacked_rnn_cells/lstm_cell/dropout_2/CastCast:rnn/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/stacked_rnn_cells/lstm_cell/dropout_2/Cast
/rnn/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul1rnn/stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:02rnn/stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1§
/rnn/stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?21
/rnn/stacked_rnn_cells/lstm_cell/dropout_3/Const
-rnn/stacked_rnn_cells/lstm_cell/dropout_3/MulMul2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:08rnn/stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/stacked_rnn_cells/lstm_cell/dropout_3/MulÄ
/rnn/stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape2rnn/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:21
/rnn/stacked_rnn_cells/lstm_cell/dropout_3/Shapeº
Frnn/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform8rnn/stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2»ý¸2H
Frnn/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform¹
8rnn/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2:
8rnn/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yÇ
6rnn/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualOrnn/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0Arnn/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6rnn/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualæ
.rnn/stacked_rnn_cells/lstm_cell/dropout_3/CastCast:rnn/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/stacked_rnn_cells/lstm_cell/dropout_3/Cast
/rnn/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul1rnn/stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:02rnn/stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Õ
#rnn/stacked_rnn_cells/lstm_cell/mulMulrnn/strided_slice_2:output:01rnn/stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn/stacked_rnn_cells/lstm_cell/mulï
5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp>rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype027
5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpõ
&rnn/stacked_rnn_cells/lstm_cell/MatMulMatMul'rnn/stacked_rnn_cells/lstm_cell/mul:z:0=rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&rnn/stacked_rnn_cells/lstm_cell/MatMulõ
7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp@rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype029
7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpæ
(rnn/stacked_rnn_cells/lstm_cell/MatMul_1MatMulrnn/zeros:output:0?rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn/stacked_rnn_cells/lstm_cell/MatMul_1ì
#rnn/stacked_rnn_cells/lstm_cell/addAddV20rnn/stacked_rnn_cells/lstm_cell/MatMul:product:02rnn/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn/stacked_rnn_cells/lstm_cell/addí
6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp?rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpù
'rnn/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd'rnn/stacked_rnn_cells/lstm_cell/add:z:0>rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell/BiasAdd¤
/rnn/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/rnn/stacked_rnn_cells/lstm_cell/split/split_dimÃ
%rnn/stacked_rnn_cells/lstm_cell/splitSplit8rnn/stacked_rnn_cells/lstm_cell/split/split_dim:output:00rnn/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2'
%rnn/stacked_rnn_cells/lstm_cell/splitÀ
'rnn/stacked_rnn_cells/lstm_cell/SigmoidSigmoid.rnn/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell/SigmoidÄ
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid.rnn/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1Í
%rnn/stacked_rnn_cells/lstm_cell/mul_1Mul-rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/mul_1·
$rnn/stacked_rnn_cells/lstm_cell/TanhTanh.rnn/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rnn/stacked_rnn_cells/lstm_cell/Tanhß
%rnn/stacked_rnn_cells/lstm_cell/mul_2Mul+rnn/stacked_rnn_cells/lstm_cell/Sigmoid:y:0(rnn/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/mul_2à
%rnn/stacked_rnn_cells/lstm_cell/add_1AddV2)rnn/stacked_rnn_cells/lstm_cell/mul_1:z:0)rnn/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/add_1Ä
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid.rnn/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2¶
&rnn/stacked_rnn_cells/lstm_cell/Tanh_1Tanh)rnn/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&rnn/stacked_rnn_cells/lstm_cell/Tanh_1ã
%rnn/stacked_rnn_cells/lstm_cell/mul_3Mul-rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0*rnn/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/mul_3¿
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape)rnn/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:23
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Shape«
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Const
+rnn/stacked_rnn_cells/lstm_cell_1/ones_likeFill:rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0:rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/stacked_rnn_cells/lstm_cell_1/ones_like§
/rnn/stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?21
/rnn/stacked_rnn_cells/lstm_cell_1/dropout/Const
-rnn/stacked_rnn_cells/lstm_cell_1/dropout/MulMul4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:08rnn/stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/stacked_rnn_cells/lstm_cell_1/dropout/MulÆ
/rnn/stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:21
/rnn/stacked_rnn_cells/lstm_cell_1/dropout/Shape¹
Frnn/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform8rnn/stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ãõ!2H
Frnn/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform¹
8rnn/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2:
8rnn/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yÇ
6rnn/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualOrnn/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0Arnn/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6rnn/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualæ
.rnn/stacked_rnn_cells/lstm_cell_1/dropout/CastCast:rnn/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/stacked_rnn_cells/lstm_cell_1/dropout/Cast
/rnn/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul1rnn/stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:02rnn/stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1«
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Const
/rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0:rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/MulÊ
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeÀ
Hrnn/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform:rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÁðÏ2J
Hrnn/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform½
:rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yÏ
8rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualQrnn/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0Crnn/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualì
0rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/CastCast<rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul3rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:04rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1«
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Const
/rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0:rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/MulÊ
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeÀ
Hrnn/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform:rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¸Ê2J
Hrnn/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform½
:rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yÏ
8rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualQrnn/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0Crnn/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualì
0rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/CastCast<rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul3rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:04rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1«
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Const
/rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0:rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/MulÊ
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape4rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeÀ
Hrnn/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform:rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¸´2J
Hrnn/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform½
:rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yÏ
8rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualQrnn/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0Crnn/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualì
0rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/CastCast<rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul3rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:04rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1è
%rnn/stacked_rnn_cells/lstm_cell_1/mulMul)rnn/stacked_rnn_cells/lstm_cell/mul_3:z:03rnn/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell_1/mulõ
7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp@rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype029
7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpý
(rnn/stacked_rnn_cells/lstm_cell_1/MatMulMatMul)rnn/stacked_rnn_cells/lstm_cell_1/mul:z:0?rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn/stacked_rnn_cells/lstm_cell_1/MatMulû
9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpBrnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02;
9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpî
*rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulrnn/zeros_2:output:0Arnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1ô
%rnn/stacked_rnn_cells/lstm_cell_1/addAddV22rnn/stacked_rnn_cells/lstm_cell_1/MatMul:product:04rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell_1/addó
8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpArnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02:
8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
)rnn/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd)rnn/stacked_rnn_cells/lstm_cell_1/add:z:0@rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd¨
1rnn/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1rnn/stacked_rnn_cells/lstm_cell_1/split/split_dimË
'rnn/stacked_rnn_cells/lstm_cell_1/splitSplit:rnn/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:02rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'rnn/stacked_rnn_cells/lstm_cell_1/splitÆ
)rnn/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid0rnn/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell_1/SigmoidÊ
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid0rnn/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ó
'rnn/stacked_rnn_cells/lstm_cell_1/mul_1Mul/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0rnn/zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/mul_1½
&rnn/stacked_rnn_cells/lstm_cell_1/TanhTanh0rnn/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&rnn/stacked_rnn_cells/lstm_cell_1/Tanhç
'rnn/stacked_rnn_cells/lstm_cell_1/mul_2Mul-rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0*rnn/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/mul_2è
'rnn/stacked_rnn_cells/lstm_cell_1/add_1AddV2+rnn/stacked_rnn_cells/lstm_cell_1/mul_1:z:0+rnn/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/add_1Ê
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid0rnn/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2¼
(rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh+rnn/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1ë
'rnn/stacked_rnn_cells/lstm_cell_1/mul_3Mul/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0,rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/mul_3
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2#
!rnn/TensorArrayV2_1/element_shapeÈ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counter¸	
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/zeros_2:output:0rnn/zeros_3:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0>rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource@rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource?rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource@rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceBrnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceArnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*#
bodyR
rnn_while_body_61367791*#
condR
rnn_while_cond_61367790*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
	rnn/while½
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   26
4rnn/TensorArrayV2Stack/TensorListStack/element_shape
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2³
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/perm¿
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn/transpose_1©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	S*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freeq
dense/Tensordot/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisï
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisõ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisÎ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat¤
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack¹
dense/Tensordot/transpose	Transposernn/transpose_1:y:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/transpose·
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/Reshape¶
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:S2
dense/Tensordot/Const_2
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisÛ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype02
dense/BiasAdd/ReadVariableOp¨
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
dense/BiasAdd¼
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup7^rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp6^rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp8^rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9^rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp8^rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:^rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp
^rnn/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2p
6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2n
5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2r
7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2t
8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2r
7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2v
9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
	rnn/while	rnn/while:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
§
while_cond_61366898
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61366898___redundant_placeholder06
2while_while_cond_61366898___redundant_placeholder16
2while_while_cond_61366898___redundant_placeholder26
2while_while_cond_61366898___redundant_placeholder36
2while_while_cond_61366898___redundant_placeholder46
2while_while_cond_61366898___redundant_placeholder56
2while_while_cond_61366898___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
·,

$__inference__traced_restore_61369711
file_prefix8
%assignvariableop_embedding_embeddings:	S2
assignvariableop_1_dense_kernel:	S+
assignvariableop_2_dense_bias:SM
9assignvariableop_3_rnn_stacked_rnn_cells_lstm_cell_kernel:
W
Cassignvariableop_4_rnn_stacked_rnn_cells_lstm_cell_recurrent_kernel:
F
7assignvariableop_5_rnn_stacked_rnn_cells_lstm_cell_bias:	O
;assignvariableop_6_rnn_stacked_rnn_cells_lstm_cell_1_kernel:
Y
Eassignvariableop_7_rnn_stacked_rnn_cells_lstm_cell_1_recurrent_kernel:
H
9assignvariableop_8_rnn_stacked_rnn_cells_lstm_cell_1_bias:	
identity_10¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8Á
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*Í
valueÃBÀ
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¢
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slicesÝ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¤
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¢
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¾
AssignVariableOp_3AssignVariableOp9assignvariableop_3_rnn_stacked_rnn_cells_lstm_cell_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4È
AssignVariableOp_4AssignVariableOpCassignvariableop_4_rnn_stacked_rnn_cells_lstm_cell_recurrent_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¼
AssignVariableOp_5AssignVariableOp7assignvariableop_5_rnn_stacked_rnn_cells_lstm_cell_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6À
AssignVariableOp_6AssignVariableOp;assignvariableop_6_rnn_stacked_rnn_cells_lstm_cell_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ê
AssignVariableOp_7AssignVariableOpEassignvariableop_7_rnn_stacked_rnn_cells_lstm_cell_1_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¾
AssignVariableOp_8AssignVariableOp9assignvariableop_8_rnn_stacked_rnn_cells_lstm_cell_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp£

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_9
Identity_10IdentityIdentity_9:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8*
T0*
_output_shapes
: 2
Identity_10"#
identity_10Identity_10:output:0*'
_input_shapes
: : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ä
'__inference_test_layer_call_fn_61366691
input_1
unknown:	S
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:	S
	unknown_7:S
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_test_layer_call_and_return_conditional_losses_613666702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
óÝ
É	
B__inference_test_layer_call_and_return_conditional_losses_61367606

inputs6
#embedding_embedding_lookup_61367334:	SR
>rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
T
@rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
N
?rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	T
@rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
V
Brnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
P
Arnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	:
'dense_tensordot_readvariableop_resource:	S3
%dense_biasadd_readvariableop_resource:S
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢embedding/embedding_lookup¢6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢	rnn/whilez
embedding/CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding/Cast½
embedding/embedding_lookupResourceGather#embedding_embedding_lookup_61367334embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding/embedding_lookup/61367334*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding/embedding_lookup¡
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding/embedding_lookup/61367334*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2%
#embedding/embedding_lookup/IdentityÈ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2'
%embedding/embedding_lookup/Identity_1t
	rnn/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
	rnn/Shape|
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice/stack
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_1
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice/stack_2ú
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slicee
rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros/mul/y|
rnn/zeros/mulMulrnn/strided_slice:output:0rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/mulg
rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros/Less/yw
rnn/zeros/LessLessrnn/zeros/mul:z:0rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros/Lessk
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros/packed/1
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros/packedg
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros/Const
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	rnn/zerosi
rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_1/mul/y
rnn/zeros_1/mulMulrnn/strided_slice:output:0rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/mulk
rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_1/Less/y
rnn/zeros_1/LessLessrnn/zeros_1/mul:z:0rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_1/Lesso
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_1/packed/1
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_1/packedk
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_1/Const
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/zeros_1i
rnn/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_2/mul/y
rnn/zeros_2/mulMulrnn/strided_slice:output:0rnn/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_2/mulk
rnn/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_2/Less/y
rnn/zeros_2/LessLessrnn/zeros_2/mul:z:0rnn/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_2/Lesso
rnn/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_2/packed/1
rnn/zeros_2/packedPackrnn/strided_slice:output:0rnn/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_2/packedk
rnn/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_2/Const
rnn/zeros_2Fillrnn/zeros_2/packed:output:0rnn/zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/zeros_2i
rnn/zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_3/mul/y
rnn/zeros_3/mulMulrnn/strided_slice:output:0rnn/zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_3/mulk
rnn/zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
rnn/zeros_3/Less/y
rnn/zeros_3/LessLessrnn/zeros_3/mul:z:0rnn/zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
rnn/zeros_3/Lesso
rnn/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
rnn/zeros_3/packed/1
rnn/zeros_3/packedPackrnn/strided_slice:output:0rnn/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn/zeros_3/packedk
rnn/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn/zeros_3/Const
rnn/zeros_3Fillrnn/zeros_3/packed:output:0rnn/zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/zeros_3}
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose/perm¸
rnn/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0rnn/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn/transpose[
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:2
rnn/Shape_1
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_1/stack
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_1
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_1/stack_2
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn/strided_slice_1
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
rnn/TensorArrayV2/element_shapeÂ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2Ç
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+rnn/TensorArrayUnstack/TensorListFromTensor
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_2/stack
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_1
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_2/stack_2
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
rnn/strided_slice_2®
/rnn/stacked_rnn_cells/lstm_cell/ones_like/ShapeShapernn/strided_slice_2:output:0*
T0*
_output_shapes
:21
/rnn/stacked_rnn_cells/lstm_cell/ones_like/Shape§
/rnn/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?21
/rnn/stacked_rnn_cells/lstm_cell/ones_like/Const
)rnn/stacked_rnn_cells/lstm_cell/ones_likeFill8rnn/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:08rnn/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell/ones_likeÖ
#rnn/stacked_rnn_cells/lstm_cell/mulMulrnn/strided_slice_2:output:02rnn/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn/stacked_rnn_cells/lstm_cell/mulï
5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp>rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype027
5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpõ
&rnn/stacked_rnn_cells/lstm_cell/MatMulMatMul'rnn/stacked_rnn_cells/lstm_cell/mul:z:0=rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&rnn/stacked_rnn_cells/lstm_cell/MatMulõ
7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp@rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype029
7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpæ
(rnn/stacked_rnn_cells/lstm_cell/MatMul_1MatMulrnn/zeros:output:0?rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn/stacked_rnn_cells/lstm_cell/MatMul_1ì
#rnn/stacked_rnn_cells/lstm_cell/addAddV20rnn/stacked_rnn_cells/lstm_cell/MatMul:product:02rnn/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#rnn/stacked_rnn_cells/lstm_cell/addí
6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp?rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpù
'rnn/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd'rnn/stacked_rnn_cells/lstm_cell/add:z:0>rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell/BiasAdd¤
/rnn/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/rnn/stacked_rnn_cells/lstm_cell/split/split_dimÃ
%rnn/stacked_rnn_cells/lstm_cell/splitSplit8rnn/stacked_rnn_cells/lstm_cell/split/split_dim:output:00rnn/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2'
%rnn/stacked_rnn_cells/lstm_cell/splitÀ
'rnn/stacked_rnn_cells/lstm_cell/SigmoidSigmoid.rnn/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell/SigmoidÄ
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid.rnn/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1Í
%rnn/stacked_rnn_cells/lstm_cell/mul_1Mul-rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/mul_1·
$rnn/stacked_rnn_cells/lstm_cell/TanhTanh.rnn/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$rnn/stacked_rnn_cells/lstm_cell/Tanhß
%rnn/stacked_rnn_cells/lstm_cell/mul_2Mul+rnn/stacked_rnn_cells/lstm_cell/Sigmoid:y:0(rnn/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/mul_2à
%rnn/stacked_rnn_cells/lstm_cell/add_1AddV2)rnn/stacked_rnn_cells/lstm_cell/mul_1:z:0)rnn/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/add_1Ä
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid.rnn/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2¶
&rnn/stacked_rnn_cells/lstm_cell/Tanh_1Tanh)rnn/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&rnn/stacked_rnn_cells/lstm_cell/Tanh_1ã
%rnn/stacked_rnn_cells/lstm_cell/mul_3Mul-rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0*rnn/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell/mul_3¿
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape)rnn/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:23
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Shape«
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Const
+rnn/stacked_rnn_cells/lstm_cell_1/ones_likeFill:rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0:rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/stacked_rnn_cells/lstm_cell_1/ones_likeé
%rnn/stacked_rnn_cells/lstm_cell_1/mulMul)rnn/stacked_rnn_cells/lstm_cell/mul_3:z:04rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell_1/mulõ
7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp@rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype029
7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpý
(rnn/stacked_rnn_cells/lstm_cell_1/MatMulMatMul)rnn/stacked_rnn_cells/lstm_cell_1/mul:z:0?rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn/stacked_rnn_cells/lstm_cell_1/MatMulû
9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpBrnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02;
9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpî
*rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulrnn/zeros_2:output:0Arnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1ô
%rnn/stacked_rnn_cells/lstm_cell_1/addAddV22rnn/stacked_rnn_cells/lstm_cell_1/MatMul:product:04rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%rnn/stacked_rnn_cells/lstm_cell_1/addó
8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpArnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02:
8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
)rnn/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd)rnn/stacked_rnn_cells/lstm_cell_1/add:z:0@rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd¨
1rnn/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1rnn/stacked_rnn_cells/lstm_cell_1/split/split_dimË
'rnn/stacked_rnn_cells/lstm_cell_1/splitSplit:rnn/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:02rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'rnn/stacked_rnn_cells/lstm_cell_1/splitÆ
)rnn/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid0rnn/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/stacked_rnn_cells/lstm_cell_1/SigmoidÊ
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid0rnn/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ó
'rnn/stacked_rnn_cells/lstm_cell_1/mul_1Mul/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0rnn/zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/mul_1½
&rnn/stacked_rnn_cells/lstm_cell_1/TanhTanh0rnn/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&rnn/stacked_rnn_cells/lstm_cell_1/Tanhç
'rnn/stacked_rnn_cells/lstm_cell_1/mul_2Mul-rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0*rnn/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/mul_2è
'rnn/stacked_rnn_cells/lstm_cell_1/add_1AddV2+rnn/stacked_rnn_cells/lstm_cell_1/mul_1:z:0+rnn/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/add_1Ê
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid0rnn/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2¼
(rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh+rnn/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1ë
'rnn/stacked_rnn_cells/lstm_cell_1/mul_3Mul/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0,rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'rnn/stacked_rnn_cells/lstm_cell_1/mul_3
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2#
!rnn/TensorArrayV2_1/element_shapeÈ
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn/TensorArrayV2_1V
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

rnn/time
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
rnn/while/maximum_iterationsr
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn/while/loop_counter¸	
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/zeros_2:output:0rnn/zeros_3:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0>rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource@rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource?rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource@rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceBrnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceArnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*#
bodyR
rnn_while_body_61367451*#
condR
rnn_while_cond_61367450*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
	rnn/while½
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   26
4rnn/TensorArrayV2Stack/TensorListStack/element_shape
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02(
&rnn/TensorArrayV2Stack/TensorListStack
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
rnn/strided_slice_3/stack
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
rnn/strided_slice_3/stack_1
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
rnn/strided_slice_3/stack_2³
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
rnn/strided_slice_3
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn/transpose_1/perm¿
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
rnn/transpose_1©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	S*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freeq
dense/Tensordot/ShapeShapernn/transpose_1:y:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisï
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisõ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisÎ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat¤
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack¹
dense/Tensordot/transpose	Transposernn/transpose_1:y:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/transpose·
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/Reshape¶
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:S2
dense/Tensordot/Const_2
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisÛ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype02
dense/BiasAdd/ReadVariableOp¨
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
dense/BiasAdd¼
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup7^rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp6^rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp8^rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9^rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp8^rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:^rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp
^rnn/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2p
6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp6rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2n
5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp5rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2r
7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp7rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2t
8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp8rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2r
7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp7rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2v
9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp9rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
	rnn/while	rnn/while:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ä
'__inference_test_layer_call_fn_61367209
input_1
unknown:	S
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:	S
	unknown_7:S
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_test_layer_call_and_return_conditional_losses_613671652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Û
î
B__inference_test_layer_call_and_return_conditional_losses_61367234
input_1%
embedding_61367212:	S 
rnn_61367215:
 
rnn_61367217:

rnn_61367219:	 
rnn_61367221:
 
rnn_61367223:

rnn_61367225:	!
dense_61367228:	S
dense_61367230:S
identity¢dense/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢rnn/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_61367212*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_613663722#
!embedding/StatefulPartitionedCallö
rnn/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0rnn_61367215rnn_61367217rnn_61367219rnn_61367221rnn_61367223rnn_61367225*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613666192
rnn/StatefulPartitionedCall¹
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_61367228dense_61367230*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_613666632
dense/StatefulPartitionedCallé
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
á

§
G__inference_embedding_layer_call_and_return_conditional_losses_61366372

inputs,
embedding_lookup_61366366:	S
identity¢embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_61366366Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/61366366*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupù
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/61366366*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identityª
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


while_body_61369134
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
X
Dwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
R
Cwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
Z
Fwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
T
Ewhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorT
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
P
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	V
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
R
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
1while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/ones_like/Shape«
1while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1while/stacked_rnn_cells/lstm_cell/ones_like/Const
+while/stacked_rnn_cells/lstm_cell/ones_likeFill:while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0:while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/ones_like§
/while/stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?21
/while/stacked_rnn_cells/lstm_cell/dropout/Const
-while/stacked_rnn_cells/lstm_cell/dropout/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:08while/stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell/dropout/MulÆ
/while/stacked_rnn_cells/lstm_cell/dropout/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:21
/while/stacked_rnn_cells/lstm_cell/dropout/Shapeº
Fwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform8while/stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¼÷2H
Fwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform¹
8while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2:
8while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yÇ
6while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualOwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0Awhile/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualæ
.while/stacked_rnn_cells/lstm_cell/dropout/CastCast:while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.while/stacked_rnn_cells/lstm_cell/dropout/Cast
/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul1while/stacked_rnn_cells/lstm_cell/dropout/Mul:z:02while/stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_1/Const
/while/stacked_rnn_cells/lstm_cell/dropout_1/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_1/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_1/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2±÷Ý2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_1/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_1/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_2/Const
/while/stacked_rnn_cells/lstm_cell/dropout_2/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_2/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_2/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2å¬Ô2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_2/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_2/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_3/Const
/while/stacked_rnn_cells/lstm_cell/dropout_3/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_3/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_3/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2»2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_3/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_3/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1ï
%while/stacked_rnn_cells/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/mul÷
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpý
(while/stacked_rnn_cells/lstm_cell/MatMulMatMul)while/stacked_rnn_cells/lstm_cell/mul:z:0?while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/MatMulý
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpí
*while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulwhile_placeholder_2Awhile/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell/MatMul_1ô
%while/stacked_rnn_cells/lstm_cell/addAddV22while/stacked_rnn_cells/lstm_cell/MatMul:product:04while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/addõ
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02:
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
)while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd)while/stacked_rnn_cells/lstm_cell/add:z:0@while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/BiasAdd¨
1while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1while/stacked_rnn_cells/lstm_cell/split/split_dimË
'while/stacked_rnn_cells/lstm_cell/splitSplit:while/stacked_rnn_cells/lstm_cell/split/split_dim:output:02while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'while/stacked_rnn_cells/lstm_cell/splitÆ
)while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid0while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/SigmoidÊ
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Ò
'while/stacked_rnn_cells/lstm_cell/mul_1Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_1½
&while/stacked_rnn_cells/lstm_cell/TanhTanh0while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/stacked_rnn_cells/lstm_cell/Tanhç
'while/stacked_rnn_cells/lstm_cell/mul_2Mul-while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0*while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_2è
'while/stacked_rnn_cells/lstm_cell/add_1AddV2+while/stacked_rnn_cells/lstm_cell/mul_1:z:0+while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/add_1Ê
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2¼
(while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh+while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/Tanh_1ë
'while/stacked_rnn_cells/lstm_cell/mul_3Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0,while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_3Å
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape+while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape¯
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Const
-while/stacked_rnn_cells/lstm_cell_1/ones_likeFill<while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0<while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/ones_like«
1while/stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell_1/dropout/Const
/while/stacked_rnn_cells/lstm_cell_1/dropout/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0:while/stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell_1/dropout/MulÌ
1while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2µ2J
Hwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell_1/dropout/CastCast<while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell_1/dropout/Cast
1while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul3while/stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:04while/stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeÆ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2»¢Ë2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_1/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeÆ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2úÌ2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_2/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeÆ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÙçÍ2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_3/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1ð
'while/stacked_rnn_cells/lstm_cell_1/mulMul+while/stacked_rnn_cells/lstm_cell/mul_3:z:05while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/mulý
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
*while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul+while/stacked_rnn_cells/lstm_cell_1/mul:z:0Awhile/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/MatMul
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpó
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulwhile_placeholder_4Cwhile/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1ü
'while/stacked_rnn_cells/lstm_cell_1/addAddV24while/stacked_rnn_cells/lstm_cell_1/MatMul:product:06while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/addû
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02<
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
+while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd+while/stacked_rnn_cells/lstm_cell_1/add:z:0Bwhile/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/BiasAdd¬
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimÓ
)while/stacked_rnn_cells/lstm_cell_1/splitSplit<while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:04while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)while/stacked_rnn_cells/lstm_cell_1/splitÌ
+while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/SigmoidÐ
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ø
)while/stacked_rnn_cells/lstm_cell_1/mul_1Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_1Ã
(while/stacked_rnn_cells/lstm_cell_1/TanhTanh2while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell_1/Tanhï
)while/stacked_rnn_cells/lstm_cell_1/mul_2Mul/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0,while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_2ð
)while/stacked_rnn_cells/lstm_cell_1/add_1AddV2-while/stacked_rnn_cells/lstm_cell_1/mul_1:z:0-while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/add_1Ð
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Â
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh-while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1ó
)while/stacked_rnn_cells/lstm_cell_1/mul_3Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0.while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_3ñ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations9^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identity+while/stacked_rnn_cells/lstm_cell/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ø
while/Identity_5Identity+while/stacked_rnn_cells/lstm_cell/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ú
while/Identity_6Identity-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6ú
while/Identity_7Identity-while/stacked_rnn_cells/lstm_cell_1/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2r
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2x
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2z
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
­
¸
rnn_while_body_61367791$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3
rnn_while_placeholder_4
rnn_while_placeholder_5#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0Z
Frnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
\
Hrnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
V
Grnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	\
Hrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
^
Jrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
X
Irnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5
rnn_while_identity_6
rnn_while_identity_7!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorX
Drnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
Z
Frnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
T
Ernn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	Z
Frnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
\
Hrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
V
Grnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpË
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeì
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItemÒ
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:27
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Shape³
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Const
/rnn/while/stacked_rnn_cells/lstm_cell/ones_likeFill>rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0>rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell/ones_like¯
3rnn/while/stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3rnn/while/stacked_rnn_cells/lstm_cell/dropout/Const
1rnn/while/stacked_rnn_cells/lstm_cell/dropout/MulMul8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0<rnn/while/stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/while/stacked_rnn_cells/lstm_cell/dropout/MulÒ
3rnn/while/stacked_rnn_cells/lstm_cell/dropout/ShapeShape8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:25
3rnn/while/stacked_rnn_cells/lstm_cell/dropout/ShapeÅ
Jrnn/while/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform<rnn/while/stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ê´`2L
Jrnn/while/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformÁ
<rnn/while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<rnn/while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y×
:rnn/while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualSrnn/while/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0Ernn/while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:rnn/while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualò
2rnn/while/stacked_rnn_cells/lstm_cell/dropout/CastCast>rnn/while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2rnn/while/stacked_rnn_cells/lstm_cell/dropout/Cast
3rnn/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul5rnn/while/stacked_rnn_cells/lstm_cell/dropout/Mul:z:06rnn/while/stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3rnn/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1³
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Const
3rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/MulMul8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0>rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/MulÖ
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/ShapeÌ
Lrnn/while/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform>rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ä2N
Lrnn/while/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformÅ
>rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2@
>rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yß
<rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualUrnn/while/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0Grnn/while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualø
4rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/CastCast@rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Cast
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul7rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:08rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1³
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Const
3rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/MulMul8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0>rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/MulÖ
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/ShapeÌ
Lrnn/while/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform>rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ùÑë2N
Lrnn/while/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformÅ
>rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2@
>rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yß
<rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualUrnn/while/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0Grnn/while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualø
4rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/CastCast@rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Cast
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul7rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:08rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1³
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Const
3rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/MulMul8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0>rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/MulÖ
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape8rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/ShapeÌ
Lrnn/while/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform>rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ýþ½2N
Lrnn/while/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformÅ
>rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2@
>rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yß
<rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualUrnn/while/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0Grnn/while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualø
4rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/CastCast@rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Cast
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul7rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:08rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5rnn/while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1ÿ
)rnn/while/stacked_rnn_cells/lstm_cell/mulMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:07rnn/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/while/stacked_rnn_cells/lstm_cell/mul
;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpFrnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp
,rnn/while/stacked_rnn_cells/lstm_cell/MatMulMatMul-rnn/while/stacked_rnn_cells/lstm_cell/mul:z:0Crnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,rnn/while/stacked_rnn_cells/lstm_cell/MatMul
=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpHrnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02?
=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpý
.rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulrnn_while_placeholder_2Ernn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1
)rnn/while/stacked_rnn_cells/lstm_cell/addAddV26rnn/while/stacked_rnn_cells/lstm_cell/MatMul:product:08rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/while/stacked_rnn_cells/lstm_cell/add
<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpGrnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02>
<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
-rnn/while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd-rnn/while/stacked_rnn_cells/lstm_cell/add:z:0Drnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd°
5rnn/while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5rnn/while/stacked_rnn_cells/lstm_cell/split/split_dimÛ
+rnn/while/stacked_rnn_cells/lstm_cell/splitSplit>rnn/while/stacked_rnn_cells/lstm_cell/split/split_dim:output:06rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2-
+rnn/while/stacked_rnn_cells/lstm_cell/splitÒ
-rnn/while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid4rnn/while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell/SigmoidÖ
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid4rnn/while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1â
+rnn/while/stacked_rnn_cells/lstm_cell/mul_1Mul3rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/mul_1É
*rnn/while/stacked_rnn_cells/lstm_cell/TanhTanh4rnn/while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*rnn/while/stacked_rnn_cells/lstm_cell/Tanh÷
+rnn/while/stacked_rnn_cells/lstm_cell/mul_2Mul1rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0.rnn/while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/mul_2ø
+rnn/while/stacked_rnn_cells/lstm_cell/add_1AddV2/rnn/while/stacked_rnn_cells/lstm_cell/mul_1:z:0/rnn/while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/add_1Ö
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid4rnn/while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2È
,rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh/rnn/while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1û
+rnn/while/stacked_rnn_cells/lstm_cell/mul_3Mul3rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:00rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/mul_3Ñ
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:29
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape·
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Const¥
1rnn/while/stacked_rnn_cells/lstm_cell_1/ones_likeFill@rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0@rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like³
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?27
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Const 
3rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/MulMul:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/MulØ
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:27
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeÌ
Lrnn/while/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ø2N
Lrnn/while/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformÅ
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2@
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yß
<rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualUrnn/while/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0Grnn/while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualø
4rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/CastCast@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Cast
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:08rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1·
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const¦
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulÜ
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeÒ
Nrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ãËÂ2P
Nrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformÉ
@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2B
@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yç
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualWrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0Irnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualþ
6rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/CastCastBrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast£
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul9rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:0:rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1·
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const¦
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulÜ
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeÒ
Nrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ö2P
Nrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformÉ
@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2B
@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yç
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualWrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0Irnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualþ
6rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/CastCastBrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast£
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul9rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:0:rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1·
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const¦
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulÜ
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeÒ
Nrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2³°2P
Nrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformÉ
@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2B
@rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yç
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualWrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0Irnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualþ
6rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/CastCastBrnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast£
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul9rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:0:rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7rnn/while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1
+rnn/while/stacked_rnn_cells/lstm_cell_1/mulMul/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:09rnn/while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell_1/mul
=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpHrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02?
=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
.rnn/while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul/rnn/while/stacked_rnn_cells/lstm_cell_1/mul:z:0Ernn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul
?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpJrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02A
?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp
0rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulrnn_while_placeholder_4Grnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1
+rnn/while/stacked_rnn_cells/lstm_cell_1/addAddV28rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul:product:0:rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell_1/add
>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpIrnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02@
>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd/rnn/while/stacked_rnn_cells/lstm_cell_1/add:z:0Frnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd´
7rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dimã
-rnn/while/stacked_rnn_cells/lstm_cell_1/splitSplit@rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:08rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/splitØ
/rnn/while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell_1/SigmoidÜ
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1è
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1Mul5rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0rnn_while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1Ï
,rnn/while/stacked_rnn_cells/lstm_cell_1/TanhTanh6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,rnn/while/stacked_rnn_cells/lstm_cell_1/Tanhÿ
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2Mul3rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:00rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2
-rnn/while/stacked_rnn_cells/lstm_cell_1/add_1AddV21rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1:z:01rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/add_1Ü
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Î
.rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh1rnn/while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3Mul5rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:02rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder1rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1ê
rnn/while/IdentityIdentityrnn/while/add_1:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_1ì
rnn/while/Identity_2Identityrnn/while/add:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_2
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_3
rnn/while/Identity_4Identity/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_4
rnn/while/Identity_5Identity/rnn/while/stacked_rnn_cells/lstm_cell/add_1:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_5
rnn/while/Identity_6Identity1rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_6
rnn/while/Identity_7Identity1rnn/while/stacked_rnn_cells/lstm_cell_1/add_1:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_7"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"5
rnn_while_identity_6rnn/while/Identity_6:output:0"5
rnn_while_identity_7rnn/while/Identity_7:output:0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"
Grnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceIrnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Hrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceJrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Frnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceHrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Ernn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceGrnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Frnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceHrnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
Drnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceFrnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"¸
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2|
<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2z
;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2~
=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2
>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2~
=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2
?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 


while_body_61366899
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
X
Dwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
R
Cwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
Z
Fwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
T
Ewhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorT
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
P
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	V
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
R
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
1while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/ones_like/Shape«
1while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1while/stacked_rnn_cells/lstm_cell/ones_like/Const
+while/stacked_rnn_cells/lstm_cell/ones_likeFill:while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0:while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/ones_like§
/while/stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?21
/while/stacked_rnn_cells/lstm_cell/dropout/Const
-while/stacked_rnn_cells/lstm_cell/dropout/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:08while/stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell/dropout/MulÆ
/while/stacked_rnn_cells/lstm_cell/dropout/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:21
/while/stacked_rnn_cells/lstm_cell/dropout/Shape¹
Fwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform8while/stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2 2H
Fwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform¹
8while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2:
8while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yÇ
6while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualOwhile/stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0Awhile/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqualæ
.while/stacked_rnn_cells/lstm_cell/dropout/CastCast:while/stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.while/stacked_rnn_cells/lstm_cell/dropout/Cast
/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul1while/stacked_rnn_cells/lstm_cell/dropout/Mul:z:02while/stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_1/Const
/while/stacked_rnn_cells/lstm_cell/dropout_1/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_1/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_1/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2£é2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_1/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_1/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_1/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_2/Const
/while/stacked_rnn_cells/lstm_cell/dropout_2/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_2/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_2/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ð2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_2/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_2/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_2/Mul_1«
1while/stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell/dropout_3/Const
/while/stacked_rnn_cells/lstm_cell/dropout_3/MulMul4while/stacked_rnn_cells/lstm_cell/ones_like:output:0:while/stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell/dropout_3/MulÊ
1while/stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape4while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/dropout_3/Shape¿
Hwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ºd2J
Hwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell/dropout_3/CastCast<while/stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell/dropout_3/Cast
1while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul3while/stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:04while/stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell/dropout_3/Mul_1ï
%while/stacked_rnn_cells/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/mul÷
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpý
(while/stacked_rnn_cells/lstm_cell/MatMulMatMul)while/stacked_rnn_cells/lstm_cell/mul:z:0?while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/MatMulý
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpí
*while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulwhile_placeholder_2Awhile/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell/MatMul_1ô
%while/stacked_rnn_cells/lstm_cell/addAddV22while/stacked_rnn_cells/lstm_cell/MatMul:product:04while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/addõ
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02:
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
)while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd)while/stacked_rnn_cells/lstm_cell/add:z:0@while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/BiasAdd¨
1while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1while/stacked_rnn_cells/lstm_cell/split/split_dimË
'while/stacked_rnn_cells/lstm_cell/splitSplit:while/stacked_rnn_cells/lstm_cell/split/split_dim:output:02while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'while/stacked_rnn_cells/lstm_cell/splitÆ
)while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid0while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/SigmoidÊ
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Ò
'while/stacked_rnn_cells/lstm_cell/mul_1Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_1½
&while/stacked_rnn_cells/lstm_cell/TanhTanh0while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/stacked_rnn_cells/lstm_cell/Tanhç
'while/stacked_rnn_cells/lstm_cell/mul_2Mul-while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0*while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_2è
'while/stacked_rnn_cells/lstm_cell/add_1AddV2+while/stacked_rnn_cells/lstm_cell/mul_1:z:0+while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/add_1Ê
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2¼
(while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh+while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/Tanh_1ë
'while/stacked_rnn_cells/lstm_cell/mul_3Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0,while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_3Å
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape+while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape¯
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Const
-while/stacked_rnn_cells/lstm_cell_1/ones_likeFill<while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0<while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/ones_like«
1while/stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?23
1while/stacked_rnn_cells/lstm_cell_1/dropout/Const
/while/stacked_rnn_cells/lstm_cell_1/dropout/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0:while/stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/while/stacked_rnn_cells/lstm_cell_1/dropout/MulÌ
1while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell_1/dropout/ShapeÀ
Hwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform:while/stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ûÈ2J
Hwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform½
:while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yÏ
8while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualQwhile/stacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0Cwhile/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualì
0while/stacked_rnn_cells/lstm_cell_1/dropout/CastCast<while/stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0while/stacked_rnn_cells/lstm_cell_1/dropout/Cast
1while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul3while/stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:04while/stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_1/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeÅ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2óÞ82L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_1/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_2/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeÆ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2þ­2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_2/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1¯
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const
1while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0<while/stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1while/stacked_rnn_cells/lstm_cell_1/dropout_3/MulÐ
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape6while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeÅ
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform<while/stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ã2L
Jwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformÁ
<while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2>
<while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y×
:while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualSwhile/stacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0Ewhile/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualò
2while/stacked_rnn_cells/lstm_cell_1/dropout_3/CastCast>while/stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul5while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:06while/stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3while/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1ð
'while/stacked_rnn_cells/lstm_cell_1/mulMul+while/stacked_rnn_cells/lstm_cell/mul_3:z:05while/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/mulý
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
*while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul+while/stacked_rnn_cells/lstm_cell_1/mul:z:0Awhile/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/MatMul
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpó
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulwhile_placeholder_4Cwhile/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1ü
'while/stacked_rnn_cells/lstm_cell_1/addAddV24while/stacked_rnn_cells/lstm_cell_1/MatMul:product:06while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/addû
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02<
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
+while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd+while/stacked_rnn_cells/lstm_cell_1/add:z:0Bwhile/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/BiasAdd¬
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimÓ
)while/stacked_rnn_cells/lstm_cell_1/splitSplit<while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:04while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)while/stacked_rnn_cells/lstm_cell_1/splitÌ
+while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/SigmoidÐ
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ø
)while/stacked_rnn_cells/lstm_cell_1/mul_1Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_1Ã
(while/stacked_rnn_cells/lstm_cell_1/TanhTanh2while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell_1/Tanhï
)while/stacked_rnn_cells/lstm_cell_1/mul_2Mul/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0,while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_2ð
)while/stacked_rnn_cells/lstm_cell_1/add_1AddV2-while/stacked_rnn_cells/lstm_cell_1/mul_1:z:0-while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/add_1Ð
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Â
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh-while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1ó
)while/stacked_rnn_cells/lstm_cell_1/mul_3Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0.while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_3ñ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations9^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identity+while/stacked_rnn_cells/lstm_cell/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ø
while/Identity_5Identity+while/stacked_rnn_cells/lstm_cell/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ú
while/Identity_6Identity-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6ú
while/Identity_7Identity-while/stacked_rnn_cells/lstm_cell_1/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2r
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2x
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2z
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
á	
ã
&__inference_signature_wrapper_61367284
input_1
unknown:	S
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:	S
	unknown_7:S
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_613653612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¶"
»
!__inference__traced_save_61369674
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableopE
Asavev2_rnn_stacked_rnn_cells_lstm_cell_kernel_read_readvariableopO
Ksavev2_rnn_stacked_rnn_cells_lstm_cell_recurrent_kernel_read_readvariableopC
?savev2_rnn_stacked_rnn_cells_lstm_cell_bias_read_readvariableopG
Csavev2_rnn_stacked_rnn_cells_lstm_cell_1_kernel_read_readvariableopQ
Msavev2_rnn_stacked_rnn_cells_lstm_cell_1_recurrent_kernel_read_readvariableopE
Asavev2_rnn_stacked_rnn_cells_lstm_cell_1_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename»
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*Í
valueÃBÀ
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slicesì
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopAsavev2_rnn_stacked_rnn_cells_lstm_cell_kernel_read_readvariableopKsavev2_rnn_stacked_rnn_cells_lstm_cell_recurrent_kernel_read_readvariableop?savev2_rnn_stacked_rnn_cells_lstm_cell_bias_read_readvariableopCsavev2_rnn_stacked_rnn_cells_lstm_cell_1_kernel_read_readvariableopMsavev2_rnn_stacked_rnn_cells_lstm_cell_1_recurrent_kernel_read_readvariableopAsavev2_rnn_stacked_rnn_cells_lstm_cell_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*q
_input_shapes`
^: :	S:	S:S:
:
::
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	S:%!

_output_shapes
:	S: 

_output_shapes
:S:&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:&"
 
_output_shapes
:
:!	

_output_shapes	
::


_output_shapes
: 
Ù¦

A__inference_rnn_layer_call_and_return_conditional_losses_61366619

inputsN
:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
P
<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
J
;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	P
<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
R
>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
L
=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	
identity¢2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
+stacked_rnn_cells/lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/ones_like/Shape
+stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+stacked_rnn_cells/lstm_cell/ones_like/Constõ
%stacked_rnn_cells/lstm_cell/ones_likeFill4stacked_rnn_cells/lstm_cell/ones_like/Shape:output:04stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/ones_likeÆ
stacked_rnn_cells/lstm_cell/mulMulstrided_slice_2:output:0.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/mulã
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpå
"stacked_rnn_cells/lstm_cell/MatMulMatMul#stacked_rnn_cells/lstm_cell/mul:z:09stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/MatMulé
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpÖ
$stacked_rnn_cells/lstm_cell/MatMul_1MatMulzeros:output:0;stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell/MatMul_1Ü
stacked_rnn_cells/lstm_cell/addAddV2,stacked_rnn_cells/lstm_cell/MatMul:product:0.stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/addá
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpé
#stacked_rnn_cells/lstm_cell/BiasAddBiasAdd#stacked_rnn_cells/lstm_cell/add:z:0:stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/BiasAdd
+stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+stacked_rnn_cells/lstm_cell/split/split_dim³
!stacked_rnn_cells/lstm_cell/splitSplit4stacked_rnn_cells/lstm_cell/split/split_dim:output:0,stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!stacked_rnn_cells/lstm_cell/split´
#stacked_rnn_cells/lstm_cell/SigmoidSigmoid*stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/Sigmoid¸
%stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid*stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_1½
!stacked_rnn_cells/lstm_cell/mul_1Mul)stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_1«
 stacked_rnn_cells/lstm_cell/TanhTanh*stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 stacked_rnn_cells/lstm_cell/TanhÏ
!stacked_rnn_cells/lstm_cell/mul_2Mul'stacked_rnn_cells/lstm_cell/Sigmoid:y:0$stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_2Ð
!stacked_rnn_cells/lstm_cell/add_1AddV2%stacked_rnn_cells/lstm_cell/mul_1:z:0%stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/add_1¸
%stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid*stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_2ª
"stacked_rnn_cells/lstm_cell/Tanh_1Tanh%stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/Tanh_1Ó
!stacked_rnn_cells/lstm_cell/mul_3Mul)stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0&stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_3³
-stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape%stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Shape£
-stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Constý
'stacked_rnn_cells/lstm_cell_1/ones_likeFill6stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:06stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/ones_likeÙ
!stacked_rnn_cells/lstm_cell_1/mulMul%stacked_rnn_cells/lstm_cell/mul_3:z:00stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/mulé
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpí
$stacked_rnn_cells/lstm_cell_1/MatMulMatMul%stacked_rnn_cells/lstm_cell_1/mul:z:0;stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/MatMulï
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype027
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÞ
&stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulzeros_2:output:0=stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stacked_rnn_cells/lstm_cell_1/MatMul_1ä
!stacked_rnn_cells/lstm_cell_1/addAddV2.stacked_rnn_cells/lstm_cell_1/MatMul:product:00stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/addç
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpñ
%stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd%stacked_rnn_cells/lstm_cell_1/add:z:0<stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/BiasAdd 
-stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-stacked_rnn_cells/lstm_cell_1/split/split_dim»
#stacked_rnn_cells/lstm_cell_1/splitSplit6stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0.stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#stacked_rnn_cells/lstm_cell_1/splitº
%stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid,stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/Sigmoid¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ã
#stacked_rnn_cells/lstm_cell_1/mul_1Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_1±
"stacked_rnn_cells/lstm_cell_1/TanhTanh,stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell_1/Tanh×
#stacked_rnn_cells/lstm_cell_1/mul_2Mul)stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0&stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_2Ø
#stacked_rnn_cells/lstm_cell_1/add_1AddV2'stacked_rnn_cells/lstm_cell_1/mul_1:z:0'stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/add_1¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2°
$stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh'stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/Tanh_1Û
#stacked_rnn_cells/lstm_cell_1/mul_3Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0(stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61366490*
condR
while_cond_61366489*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1½
IdentityIdentitytranspose_1:y:03^stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2^stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp4^stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp5^stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4^stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp6^stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2h
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2f
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2l
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2n
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë

rnn_while_cond_61367450$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3
rnn_while_placeholder_4
rnn_while_placeholder_5&
"rnn_while_less_rnn_strided_slice_1>
:rnn_while_rnn_while_cond_61367450___redundant_placeholder0>
:rnn_while_rnn_while_cond_61367450___redundant_placeholder1>
:rnn_while_rnn_while_cond_61367450___redundant_placeholder2>
:rnn_while_rnn_while_cond_61367450___redundant_placeholder3>
:rnn_while_rnn_while_cond_61367450___redundant_placeholder4>
:rnn_while_rnn_while_cond_61367450___redundant_placeholder5>
:rnn_while_rnn_while_cond_61367450___redundant_placeholder6
rnn_while_identity

rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
µ

while_body_61368210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
X
Dwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
R
Cwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
Z
Fwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
T
Ewhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorT
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
P
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	V
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
R
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
1while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/ones_like/Shape«
1while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1while/stacked_rnn_cells/lstm_cell/ones_like/Const
+while/stacked_rnn_cells/lstm_cell/ones_likeFill:while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0:while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/ones_likeð
%while/stacked_rnn_cells/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:04while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/mul÷
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpý
(while/stacked_rnn_cells/lstm_cell/MatMulMatMul)while/stacked_rnn_cells/lstm_cell/mul:z:0?while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/MatMulý
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpí
*while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulwhile_placeholder_2Awhile/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell/MatMul_1ô
%while/stacked_rnn_cells/lstm_cell/addAddV22while/stacked_rnn_cells/lstm_cell/MatMul:product:04while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/addõ
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02:
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
)while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd)while/stacked_rnn_cells/lstm_cell/add:z:0@while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/BiasAdd¨
1while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1while/stacked_rnn_cells/lstm_cell/split/split_dimË
'while/stacked_rnn_cells/lstm_cell/splitSplit:while/stacked_rnn_cells/lstm_cell/split/split_dim:output:02while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'while/stacked_rnn_cells/lstm_cell/splitÆ
)while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid0while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/SigmoidÊ
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Ò
'while/stacked_rnn_cells/lstm_cell/mul_1Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_1½
&while/stacked_rnn_cells/lstm_cell/TanhTanh0while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/stacked_rnn_cells/lstm_cell/Tanhç
'while/stacked_rnn_cells/lstm_cell/mul_2Mul-while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0*while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_2è
'while/stacked_rnn_cells/lstm_cell/add_1AddV2+while/stacked_rnn_cells/lstm_cell/mul_1:z:0+while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/add_1Ê
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2¼
(while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh+while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/Tanh_1ë
'while/stacked_rnn_cells/lstm_cell/mul_3Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0,while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_3Å
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape+while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape¯
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Const
-while/stacked_rnn_cells/lstm_cell_1/ones_likeFill<while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0<while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/ones_likeñ
'while/stacked_rnn_cells/lstm_cell_1/mulMul+while/stacked_rnn_cells/lstm_cell/mul_3:z:06while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/mulý
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
*while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul+while/stacked_rnn_cells/lstm_cell_1/mul:z:0Awhile/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/MatMul
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpó
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulwhile_placeholder_4Cwhile/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1ü
'while/stacked_rnn_cells/lstm_cell_1/addAddV24while/stacked_rnn_cells/lstm_cell_1/MatMul:product:06while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/addû
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02<
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
+while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd+while/stacked_rnn_cells/lstm_cell_1/add:z:0Bwhile/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/BiasAdd¬
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimÓ
)while/stacked_rnn_cells/lstm_cell_1/splitSplit<while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:04while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)while/stacked_rnn_cells/lstm_cell_1/splitÌ
+while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/SigmoidÐ
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ø
)while/stacked_rnn_cells/lstm_cell_1/mul_1Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_1Ã
(while/stacked_rnn_cells/lstm_cell_1/TanhTanh2while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell_1/Tanhï
)while/stacked_rnn_cells/lstm_cell_1/mul_2Mul/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0,while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_2ð
)while/stacked_rnn_cells/lstm_cell_1/add_1AddV2-while/stacked_rnn_cells/lstm_cell_1/mul_1:z:0-while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/add_1Ð
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Â
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh-while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1ó
)while/stacked_rnn_cells/lstm_cell_1/mul_3Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0.while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_3ñ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations9^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identity+while/stacked_rnn_cells/lstm_cell/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ø
while/Identity_5Identity+while/stacked_rnn_cells/lstm_cell/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ú
while/Identity_6Identity-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6ú
while/Identity_7Identity-while/stacked_rnn_cells/lstm_cell_1/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2r
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2x
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2z
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
Ãµ

test_rnn_while_body_61365206.
*test_rnn_while_test_rnn_while_loop_counter4
0test_rnn_while_test_rnn_while_maximum_iterations
test_rnn_while_placeholder 
test_rnn_while_placeholder_1 
test_rnn_while_placeholder_2 
test_rnn_while_placeholder_3 
test_rnn_while_placeholder_4 
test_rnn_while_placeholder_5-
)test_rnn_while_test_rnn_strided_slice_1_0i
etest_rnn_while_tensorarrayv2read_tensorlistgetitem_test_rnn_tensorarrayunstack_tensorlistfromtensor_0_
Ktest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
a
Mtest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
[
Ltest_rnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	a
Mtest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
c
Otest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
]
Ntest_rnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
test_rnn_while_identity
test_rnn_while_identity_1
test_rnn_while_identity_2
test_rnn_while_identity_3
test_rnn_while_identity_4
test_rnn_while_identity_5
test_rnn_while_identity_6
test_rnn_while_identity_7+
'test_rnn_while_test_rnn_strided_slice_1g
ctest_rnn_while_tensorarrayv2read_tensorlistgetitem_test_rnn_tensorarrayunstack_tensorlistfromtensor]
Itest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
_
Ktest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
Y
Jtest_rnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	_
Ktest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
a
Mtest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
[
Ltest_rnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢Atest/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢@test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢Btest/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢Ctest/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢Btest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢Dtest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÕ
@test/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2B
@test/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape
2test/rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemetest_rnn_while_tensorarrayv2read_tensorlistgetitem_test_rnn_tensorarrayunstack_tensorlistfromtensor_0test_rnn_while_placeholderItest/rnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype024
2test/rnn/while/TensorArrayV2Read/TensorListGetItemá
:test/rnn/while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape9test/rnn/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2<
:test/rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Shape½
:test/rnn/while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:test/rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Const±
4test/rnn/while/stacked_rnn_cells/lstm_cell/ones_likeFillCtest/rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0Ctest/rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4test/rnn/while/stacked_rnn_cells/lstm_cell/ones_like
.test/rnn/while/stacked_rnn_cells/lstm_cell/mulMul9test/rnn/while/TensorArrayV2Read/TensorListGetItem:item:0=test/rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.test/rnn/while/stacked_rnn_cells/lstm_cell/mul
@test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpKtest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02B
@test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¡
1test/rnn/while/stacked_rnn_cells/lstm_cell/MatMulMatMul2test/rnn/while/stacked_rnn_cells/lstm_cell/mul:z:0Htest/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul
Btest/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpMtest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02D
Btest/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp
3test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1MatMultest_rnn_while_placeholder_2Jtest/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1
.test/rnn/while/stacked_rnn_cells/lstm_cell/addAddV2;test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul:product:0=test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.test/rnn/while/stacked_rnn_cells/lstm_cell/add
Atest/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpLtest_rnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02C
Atest/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¥
2test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd2test/rnn/while/stacked_rnn_cells/lstm_cell/add:z:0Itest/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAddº
:test/rnn/while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:test/rnn/while/stacked_rnn_cells/lstm_cell/split/split_dimï
0test/rnn/while/stacked_rnn_cells/lstm_cell/splitSplitCtest/rnn/while/stacked_rnn_cells/lstm_cell/split/split_dim:output:0;test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split22
0test/rnn/while/stacked_rnn_cells/lstm_cell/splitá
2test/rnn/while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid9test/rnn/while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoidå
4test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid9test/rnn/while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1ö
0test/rnn/while/stacked_rnn_cells/lstm_cell/mul_1Mul8test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0test_rnn_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/while/stacked_rnn_cells/lstm_cell/mul_1Ø
/test/rnn/while/stacked_rnn_cells/lstm_cell/TanhTanh9test/rnn/while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/test/rnn/while/stacked_rnn_cells/lstm_cell/Tanh
0test/rnn/while/stacked_rnn_cells/lstm_cell/mul_2Mul6test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid:y:03test/rnn/while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/while/stacked_rnn_cells/lstm_cell/mul_2
0test/rnn/while/stacked_rnn_cells/lstm_cell/add_1AddV24test/rnn/while/stacked_rnn_cells/lstm_cell/mul_1:z:04test/rnn/while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/while/stacked_rnn_cells/lstm_cell/add_1å
4test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid9test/rnn/while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2×
1test/rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh4test/rnn/while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1test/rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1
0test/rnn/while/stacked_rnn_cells/lstm_cell/mul_3Mul8test/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:05test/rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/while/stacked_rnn_cells/lstm_cell/mul_3à
<test/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape4test/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:2>
<test/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeÁ
<test/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2>
<test/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Const¹
6test/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_likeFillEtest/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0Etest/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6test/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like
0test/rnn/while/stacked_rnn_cells/lstm_cell_1/mulMul4test/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0?test/rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul
Btest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpMtest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02D
Btest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp©
3test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul4test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul:z:0Jtest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul
Dtest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpOtest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02F
Dtest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp
5test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMultest_rnn_while_placeholder_4Ltest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1 
0test/rnn/while/stacked_rnn_cells/lstm_cell_1/addAddV2=test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul:product:0?test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/while/stacked_rnn_cells/lstm_cell_1/add
Ctest/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpNtest_rnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02E
Ctest/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp­
4test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd4test/rnn/while/stacked_rnn_cells/lstm_cell_1/add:z:0Ktest/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd¾
<test/rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2>
<test/rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dim÷
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/splitSplitEtest/rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0=test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split24
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/splitç
4test/rnn/while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid;test/rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoidë
6test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid;test/rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1ü
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1Mul:test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0test_rnn_while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1Þ
1test/rnn/while/stacked_rnn_cells/lstm_cell_1/TanhTanh;test/rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1test/rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2Mul8test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:05test/rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/add_1AddV26test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1:z:06test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/add_1ë
6test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid;test/rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Ý
3test/rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh6test/rnn/while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3test/rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3Mul:test/rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:07test/rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3
3test/rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemtest_rnn_while_placeholder_1test_rnn_while_placeholder6test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype025
3test/rnn/while/TensorArrayV2Write/TensorListSetItemn
test/rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
test/rnn/while/add/y
test/rnn/while/addAddV2test_rnn_while_placeholdertest/rnn/while/add/y:output:0*
T0*
_output_shapes
: 2
test/rnn/while/addr
test/rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
test/rnn/while/add_1/y£
test/rnn/while/add_1AddV2*test_rnn_while_test_rnn_while_loop_countertest/rnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
test/rnn/while/add_1
test/rnn/while/IdentityIdentitytest/rnn/while/add_1:z:0B^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
test/rnn/while/Identity³
test/rnn/while/Identity_1Identity0test_rnn_while_test_rnn_while_maximum_iterationsB^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
test/rnn/while/Identity_1
test/rnn/while/Identity_2Identitytest/rnn/while/add:z:0B^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
test/rnn/while/Identity_2Æ
test/rnn/while/Identity_3IdentityCtest/rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0B^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
test/rnn/while/Identity_3É
test/rnn/while/Identity_4Identity4test/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0B^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/while/Identity_4É
test/rnn/while/Identity_5Identity4test/rnn/while/stacked_rnn_cells/lstm_cell/add_1:z:0B^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/while/Identity_5Ë
test/rnn/while/Identity_6Identity6test/rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0B^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/while/Identity_6Ë
test/rnn/while/Identity_7Identity6test/rnn/while/stacked_rnn_cells/lstm_cell_1/add_1:z:0B^test/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpA^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpD^test/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpC^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpE^test/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/while/Identity_7";
test_rnn_while_identity test/rnn/while/Identity:output:0"?
test_rnn_while_identity_1"test/rnn/while/Identity_1:output:0"?
test_rnn_while_identity_2"test/rnn/while/Identity_2:output:0"?
test_rnn_while_identity_3"test/rnn/while/Identity_3:output:0"?
test_rnn_while_identity_4"test/rnn/while/Identity_4:output:0"?
test_rnn_while_identity_5"test/rnn/while/Identity_5:output:0"?
test_rnn_while_identity_6"test/rnn/while/Identity_6:output:0"?
test_rnn_while_identity_7"test/rnn/while/Identity_7:output:0"
Ltest_rnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceNtest_rnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0" 
Mtest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceOtest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Ktest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceMtest_rnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Jtest_rnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceLtest_rnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Ktest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceMtest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
Itest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceKtest_rnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"Ì
ctest_rnn_while_tensorarrayv2read_tensorlistgetitem_test_rnn_tensorarrayunstack_tensorlistfromtensoretest_rnn_while_tensorarrayv2read_tensorlistgetitem_test_rnn_tensorarrayunstack_tensorlistfromtensor_0"T
'test_rnn_while_test_rnn_strided_slice_1)test_rnn_while_test_rnn_strided_slice_1_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2
Atest/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpAtest/rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2
@test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp@test/rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2
Btest/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpBtest/rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2
Ctest/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpCtest/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2
Btest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpBtest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2
Dtest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpDtest/rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
¼P
²
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61365488

inputs

states
states_1
states_2
states_3<
(lstm_cell_matmul_readvariableop_resource:
>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	>
*lstm_cell_1_matmul_readvariableop_resource:
@
,lstm_cell_1_matmul_1_readvariableop_resource:
:
+lstm_cell_1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3

identity_4¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢"lstm_cell_1/BiasAdd/ReadVariableOp¢!lstm_cell_1/MatMul/ReadVariableOp¢#lstm_cell_1/MatMul_1/ReadVariableOpl
lstm_cell/ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/ones_like~
lstm_cell/mulMulinputslstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMullstm_cell/mul:z:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul³
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulstates)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add«
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp¡
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimë
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell/split~
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_1
lstm_cell/mul_1Mullstm_cell/Sigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_1u
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh
lstm_cell/mul_2Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2
lstm_cell/add_1AddV2lstm_cell/mul_1:z:0lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh_1
lstm_cell/mul_3Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_3}
lstm_cell_1/ones_like/ShapeShapelstm_cell/mul_3:z:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_like
lstm_cell_1/mulMullstm_cell/mul_3:z:0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul³
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell_1/MatMul/ReadVariableOp¥
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul¹
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_cell_1/MatMul_1/ReadVariableOp 
lstm_cell_1/MatMul_1MatMulstates_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add±
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/BiasAdd/ReadVariableOp©
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dimó
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_1/split
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid_1:y:0states_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1{
lstm_cell_1/TanhTanhlstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/add_1AddV2lstm_cell_1/mul_1:z:0lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_3Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3Â
IdentityIdentitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÄ

Identity_1Identitylstm_cell/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ä

Identity_2Identitylstm_cell/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2Æ

Identity_3Identitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3Æ

Identity_4Identitylstm_cell_1/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ò
§
while_cond_61365900
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61365900___redundant_placeholder06
2while_while_cond_61365900___redundant_placeholder16
2while_while_cond_61365900___redundant_placeholder26
2while_while_cond_61365900___redundant_placeholder36
2while_while_cond_61365900___redundant_placeholder46
2while_while_cond_61365900___redundant_placeholder56
2while_while_cond_61365900___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
¨3
¶

while_body_61365901
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
"while_stacked_rnn_cells_61365929_0:
6
"while_stacked_rnn_cells_61365931_0:
1
"while_stacked_rnn_cells_61365933_0:	6
"while_stacked_rnn_cells_61365935_0:
6
"while_stacked_rnn_cells_61365937_0:
1
"while_stacked_rnn_cells_61365939_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
 while_stacked_rnn_cells_61365929:
4
 while_stacked_rnn_cells_61365931:
/
 while_stacked_rnn_cells_61365933:	4
 while_stacked_rnn_cells_61365935:
4
 while_stacked_rnn_cells_61365937:
/
 while_stacked_rnn_cells_61365939:	¢/while/stacked_rnn_cells/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÕ
/while/stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_placeholder_4while_placeholder_5"while_stacked_rnn_cells_61365929_0"while_stacked_rnn_cells_61365931_0"while_stacked_rnn_cells_61365933_0"while_stacked_rnn_cells_61365935_0"while_stacked_rnn_cells_61365937_0"while_stacked_rnn_cells_61365939_0*
Tin
2*
Tout	
2*
_collective_manager_ids
 *x
_output_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_6136578321
/while/stacked_rnn_cells/StatefulPartitionedCallü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/stacked_rnn_cells/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:00^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity£
while/Identity_1Identitywhile_while_maximum_iterations0^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:00^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¿
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ï
while/Identity_4Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:10^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4Ï
while/Identity_5Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:20^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Ï
while/Identity_6Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:30^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6Ï
while/Identity_7Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:40^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"F
 while_stacked_rnn_cells_61365929"while_stacked_rnn_cells_61365929_0"F
 while_stacked_rnn_cells_61365931"while_stacked_rnn_cells_61365931_0"F
 while_stacked_rnn_cells_61365933"while_stacked_rnn_cells_61365933_0"F
 while_stacked_rnn_cells_61365935"while_stacked_rnn_cells_61365935_0"F
 while_stacked_rnn_cells_61365937"while_stacked_rnn_cells_61365937_0"F
 while_stacked_rnn_cells_61365939"while_stacked_rnn_cells_61365939_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2b
/while/stacked_rnn_cells/StatefulPartitionedCall/while/stacked_rnn_cells/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
	

&__inference_rnn_layer_call_fn_61368044
inputs_0
unknown:

	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613656002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ë

rnn_while_cond_61367790$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3
rnn_while_placeholder_4
rnn_while_placeholder_5&
"rnn_while_less_rnn_strided_slice_1>
:rnn_while_rnn_while_cond_61367790___redundant_placeholder0>
:rnn_while_rnn_while_cond_61367790___redundant_placeholder1>
:rnn_while_rnn_while_cond_61367790___redundant_placeholder2>
:rnn_while_rnn_while_cond_61367790___redundant_placeholder3>
:rnn_while_rnn_while_cond_61367790___redundant_placeholder4>
:rnn_while_rnn_while_cond_61367790___redundant_placeholder5>
:rnn_while_rnn_while_cond_61367790___redundant_placeholder6
rnn_while_identity

rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: 2
rnn/while/Lessi
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn/while/Identity"1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
¨3
¶

while_body_61365510
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
"while_stacked_rnn_cells_61365538_0:
6
"while_stacked_rnn_cells_61365540_0:
1
"while_stacked_rnn_cells_61365542_0:	6
"while_stacked_rnn_cells_61365544_0:
6
"while_stacked_rnn_cells_61365546_0:
1
"while_stacked_rnn_cells_61365548_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
 while_stacked_rnn_cells_61365538:
4
 while_stacked_rnn_cells_61365540:
/
 while_stacked_rnn_cells_61365542:	4
 while_stacked_rnn_cells_61365544:
4
 while_stacked_rnn_cells_61365546:
/
 while_stacked_rnn_cells_61365548:	¢/while/stacked_rnn_cells/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÕ
/while/stacked_rnn_cells/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_placeholder_4while_placeholder_5"while_stacked_rnn_cells_61365538_0"while_stacked_rnn_cells_61365540_0"while_stacked_rnn_cells_61365542_0"while_stacked_rnn_cells_61365544_0"while_stacked_rnn_cells_61365546_0"while_stacked_rnn_cells_61365548_0*
Tin
2*
Tout	
2*
_collective_manager_ids
 *x
_output_shapesf
d:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_6136548821
/while/stacked_rnn_cells/StatefulPartitionedCallü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/stacked_rnn_cells/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:00^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity£
while/Identity_1Identitywhile_while_maximum_iterations0^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:00^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¿
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:00^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3Ï
while/Identity_4Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:10^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4Ï
while/Identity_5Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:20^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Ï
while/Identity_6Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:30^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6Ï
while/Identity_7Identity8while/stacked_rnn_cells/StatefulPartitionedCall:output:40^while/stacked_rnn_cells/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"F
 while_stacked_rnn_cells_61365538"while_stacked_rnn_cells_61365538_0"F
 while_stacked_rnn_cells_61365540"while_stacked_rnn_cells_61365540_0"F
 while_stacked_rnn_cells_61365542"while_stacked_rnn_cells_61365542_0"F
 while_stacked_rnn_cells_61365544"while_stacked_rnn_cells_61365544_0"F
 while_stacked_rnn_cells_61365546"while_stacked_rnn_cells_61365546_0"F
 while_stacked_rnn_cells_61365548"while_stacked_rnn_cells_61365548_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2b
/while/stacked_rnn_cells/StatefulPartitionedCall/while/stacked_rnn_cells/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
áP
¼
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61369492

inputs

states_0_0

states_0_1

states_1_0

states_1_1<
(lstm_cell_matmul_readvariableop_resource:
>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	>
*lstm_cell_1_matmul_readvariableop_resource:
@
,lstm_cell_1_matmul_1_readvariableop_resource:
:
+lstm_cell_1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3

identity_4¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢"lstm_cell_1/BiasAdd/ReadVariableOp¢!lstm_cell_1/MatMul/ReadVariableOp¢#lstm_cell_1/MatMul_1/ReadVariableOpl
lstm_cell/ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/ones_like~
lstm_cell/mulMulinputslstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMullstm_cell/mul:z:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul³
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMul
states_0_0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add«
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp¡
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimë
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell/split~
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_1
lstm_cell/mul_1Mullstm_cell/Sigmoid_1:y:0
states_0_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_1u
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh
lstm_cell/mul_2Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2
lstm_cell/add_1AddV2lstm_cell/mul_1:z:0lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh_1
lstm_cell/mul_3Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_3}
lstm_cell_1/ones_like/ShapeShapelstm_cell/mul_3:z:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_like
lstm_cell_1/mulMullstm_cell/mul_3:z:0lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul³
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell_1/MatMul/ReadVariableOp¥
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul¹
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_cell_1/MatMul_1/ReadVariableOp¢
lstm_cell_1/MatMul_1MatMul
states_1_0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add±
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/BiasAdd/ReadVariableOp©
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dimó
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_1/split
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid_1:y:0
states_1_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1{
lstm_cell_1/TanhTanhlstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/add_1AddV2lstm_cell_1/mul_1:z:0lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_3Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3Â
IdentityIdentitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÄ

Identity_1Identitylstm_cell/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ä

Identity_2Identitylstm_cell/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2Æ

Identity_3Identitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3Æ

Identity_4Identitylstm_cell_1/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/1
µ

while_body_61366490
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
X
Dwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
R
Cwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
Z
Fwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
T
Ewhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorT
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
P
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	V
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
R
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
1while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/ones_like/Shape«
1while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1while/stacked_rnn_cells/lstm_cell/ones_like/Const
+while/stacked_rnn_cells/lstm_cell/ones_likeFill:while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0:while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/ones_likeð
%while/stacked_rnn_cells/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:04while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/mul÷
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpý
(while/stacked_rnn_cells/lstm_cell/MatMulMatMul)while/stacked_rnn_cells/lstm_cell/mul:z:0?while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/MatMulý
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpí
*while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulwhile_placeholder_2Awhile/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell/MatMul_1ô
%while/stacked_rnn_cells/lstm_cell/addAddV22while/stacked_rnn_cells/lstm_cell/MatMul:product:04while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/addõ
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02:
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
)while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd)while/stacked_rnn_cells/lstm_cell/add:z:0@while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/BiasAdd¨
1while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1while/stacked_rnn_cells/lstm_cell/split/split_dimË
'while/stacked_rnn_cells/lstm_cell/splitSplit:while/stacked_rnn_cells/lstm_cell/split/split_dim:output:02while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'while/stacked_rnn_cells/lstm_cell/splitÆ
)while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid0while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/SigmoidÊ
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Ò
'while/stacked_rnn_cells/lstm_cell/mul_1Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_1½
&while/stacked_rnn_cells/lstm_cell/TanhTanh0while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/stacked_rnn_cells/lstm_cell/Tanhç
'while/stacked_rnn_cells/lstm_cell/mul_2Mul-while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0*while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_2è
'while/stacked_rnn_cells/lstm_cell/add_1AddV2+while/stacked_rnn_cells/lstm_cell/mul_1:z:0+while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/add_1Ê
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2¼
(while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh+while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/Tanh_1ë
'while/stacked_rnn_cells/lstm_cell/mul_3Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0,while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_3Å
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape+while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape¯
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Const
-while/stacked_rnn_cells/lstm_cell_1/ones_likeFill<while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0<while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/ones_likeñ
'while/stacked_rnn_cells/lstm_cell_1/mulMul+while/stacked_rnn_cells/lstm_cell/mul_3:z:06while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/mulý
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
*while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul+while/stacked_rnn_cells/lstm_cell_1/mul:z:0Awhile/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/MatMul
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpó
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulwhile_placeholder_4Cwhile/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1ü
'while/stacked_rnn_cells/lstm_cell_1/addAddV24while/stacked_rnn_cells/lstm_cell_1/MatMul:product:06while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/addû
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02<
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
+while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd+while/stacked_rnn_cells/lstm_cell_1/add:z:0Bwhile/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/BiasAdd¬
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimÓ
)while/stacked_rnn_cells/lstm_cell_1/splitSplit<while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:04while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)while/stacked_rnn_cells/lstm_cell_1/splitÌ
+while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/SigmoidÐ
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ø
)while/stacked_rnn_cells/lstm_cell_1/mul_1Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_1Ã
(while/stacked_rnn_cells/lstm_cell_1/TanhTanh2while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell_1/Tanhï
)while/stacked_rnn_cells/lstm_cell_1/mul_2Mul/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0,while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_2ð
)while/stacked_rnn_cells/lstm_cell_1/add_1AddV2-while/stacked_rnn_cells/lstm_cell_1/mul_1:z:0-while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/add_1Ð
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Â
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh-while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1ó
)while/stacked_rnn_cells/lstm_cell_1/mul_3Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0.while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_3ñ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations9^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identity+while/stacked_rnn_cells/lstm_cell/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ø
while/Identity_5Identity+while/stacked_rnn_cells/lstm_cell/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ú
while/Identity_6Identity-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6ú
while/Identity_7Identity-while/stacked_rnn_cells/lstm_cell_1/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2r
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2x
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2z
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
µ

while_body_61368826
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
X
Dwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
R
Cwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
Z
Fwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
T
Ewhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_identity_6
while_identity_7
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorT
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
V
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
P
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	V
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
X
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
R
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
1while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:23
1while/stacked_rnn_cells/lstm_cell/ones_like/Shape«
1while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1while/stacked_rnn_cells/lstm_cell/ones_like/Const
+while/stacked_rnn_cells/lstm_cell/ones_likeFill:while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0:while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/ones_likeð
%while/stacked_rnn_cells/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:04while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/mul÷
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype029
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpý
(while/stacked_rnn_cells/lstm_cell/MatMulMatMul)while/stacked_rnn_cells/lstm_cell/mul:z:0?while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/MatMulý
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpí
*while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulwhile_placeholder_2Awhile/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell/MatMul_1ô
%while/stacked_rnn_cells/lstm_cell/addAddV22while/stacked_rnn_cells/lstm_cell/MatMul:product:04while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%while/stacked_rnn_cells/lstm_cell/addõ
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02:
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
)while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd)while/stacked_rnn_cells/lstm_cell/add:z:0@while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/BiasAdd¨
1while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1while/stacked_rnn_cells/lstm_cell/split/split_dimË
'while/stacked_rnn_cells/lstm_cell/splitSplit:while/stacked_rnn_cells/lstm_cell/split/split_dim:output:02while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'while/stacked_rnn_cells/lstm_cell/splitÆ
)while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid0while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell/SigmoidÊ
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_1Ò
'while/stacked_rnn_cells/lstm_cell/mul_1Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_1½
&while/stacked_rnn_cells/lstm_cell/TanhTanh0while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&while/stacked_rnn_cells/lstm_cell/Tanhç
'while/stacked_rnn_cells/lstm_cell/mul_2Mul-while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0*while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_2è
'while/stacked_rnn_cells/lstm_cell/add_1AddV2+while/stacked_rnn_cells/lstm_cell/mul_1:z:0+while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/add_1Ê
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid0while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell/Sigmoid_2¼
(while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh+while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell/Tanh_1ë
'while/stacked_rnn_cells/lstm_cell/mul_3Mul/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0,while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell/mul_3Å
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape+while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape¯
3while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3while/stacked_rnn_cells/lstm_cell_1/ones_like/Const
-while/stacked_rnn_cells/lstm_cell_1/ones_likeFill<while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0<while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/ones_likeñ
'while/stacked_rnn_cells/lstm_cell_1/mulMul+while/stacked_rnn_cells/lstm_cell/mul_3:z:06while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/mulý
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02;
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
*while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul+while/stacked_rnn_cells/lstm_cell_1/mul:z:0Awhile/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/MatMul
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpó
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulwhile_placeholder_4Cwhile/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,while/stacked_rnn_cells/lstm_cell_1/MatMul_1ü
'while/stacked_rnn_cells/lstm_cell_1/addAddV24while/stacked_rnn_cells/lstm_cell_1/MatMul:product:06while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'while/stacked_rnn_cells/lstm_cell_1/addû
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02<
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
+while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd+while/stacked_rnn_cells/lstm_cell_1/add:z:0Bwhile/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/BiasAdd¬
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :25
3while/stacked_rnn_cells/lstm_cell_1/split/split_dimÓ
)while/stacked_rnn_cells/lstm_cell_1/splitSplit<while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:04while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2+
)while/stacked_rnn_cells/lstm_cell_1/splitÌ
+while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+while/stacked_rnn_cells/lstm_cell_1/SigmoidÐ
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ø
)while/stacked_rnn_cells/lstm_cell_1/mul_1Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_1Ã
(while/stacked_rnn_cells/lstm_cell_1/TanhTanh2while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(while/stacked_rnn_cells/lstm_cell_1/Tanhï
)while/stacked_rnn_cells/lstm_cell_1/mul_2Mul/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0,while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_2ð
)while/stacked_rnn_cells/lstm_cell_1/add_1AddV2-while/stacked_rnn_cells/lstm_cell_1/mul_1:z:0-while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/add_1Ð
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid2while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Â
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh-while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*while/stacked_rnn_cells/lstm_cell_1/Tanh_1ó
)while/stacked_rnn_cells/lstm_cell_1/mul_3Mul1while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0.while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)while/stacked_rnn_cells/lstm_cell_1/mul_3ñ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1Æ
while/IdentityIdentitywhile/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/IdentityÙ
while/Identity_1Identitywhile_while_maximum_iterations9^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1È
while/Identity_2Identitywhile/add:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2õ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3ø
while/Identity_4Identity+while/stacked_rnn_cells/lstm_cell/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4ø
while/Identity_5Identity+while/stacked_rnn_cells/lstm_cell/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5ú
while/Identity_6Identity-while/stacked_rnn_cells/lstm_cell_1/mul_3:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_6ú
while/Identity_7Identity-while/stacked_rnn_cells/lstm_cell_1/add_1:z:09^while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8^while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp;^while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:^while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<^while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_7")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"-
while_identity_6while/Identity_6:output:0"-
while_identity_7while/Identity_7:output:0"
Cwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceEwhile_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Dwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceFwhile_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Awhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceCwhile_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Bwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceDwhile_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
@while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceBwhile_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp8while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2r
7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp7while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp9while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2x
:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2v
9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp9while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2z
;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp;while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
Ò
§
while_cond_61368517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61368517___redundant_placeholder06
2while_while_cond_61368517___redundant_placeholder16
2while_while_cond_61368517___redundant_placeholder26
2while_while_cond_61368517___redundant_placeholder36
2while_while_cond_61368517___redundant_placeholder46
2while_while_cond_61368517___redundant_placeholder56
2while_while_cond_61368517___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:
É 
û
C__inference_dense_layer_call_and_return_conditional_losses_61366663

inputs4
!tensordot_readvariableop_resource:	S-
biasadd_readvariableop_resource:S
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	S*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:S2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2	
BiasAdd¥
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á¦

A__inference_rnn_layer_call_and_return_conditional_losses_61368339
inputs_0N
:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
P
<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
J
;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	P
<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
R
>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
L
=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	
identity¢2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
+stacked_rnn_cells/lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/ones_like/Shape
+stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+stacked_rnn_cells/lstm_cell/ones_like/Constõ
%stacked_rnn_cells/lstm_cell/ones_likeFill4stacked_rnn_cells/lstm_cell/ones_like/Shape:output:04stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/ones_likeÆ
stacked_rnn_cells/lstm_cell/mulMulstrided_slice_2:output:0.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/mulã
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpå
"stacked_rnn_cells/lstm_cell/MatMulMatMul#stacked_rnn_cells/lstm_cell/mul:z:09stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/MatMulé
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpÖ
$stacked_rnn_cells/lstm_cell/MatMul_1MatMulzeros:output:0;stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell/MatMul_1Ü
stacked_rnn_cells/lstm_cell/addAddV2,stacked_rnn_cells/lstm_cell/MatMul:product:0.stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/addá
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpé
#stacked_rnn_cells/lstm_cell/BiasAddBiasAdd#stacked_rnn_cells/lstm_cell/add:z:0:stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/BiasAdd
+stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+stacked_rnn_cells/lstm_cell/split/split_dim³
!stacked_rnn_cells/lstm_cell/splitSplit4stacked_rnn_cells/lstm_cell/split/split_dim:output:0,stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!stacked_rnn_cells/lstm_cell/split´
#stacked_rnn_cells/lstm_cell/SigmoidSigmoid*stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/Sigmoid¸
%stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid*stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_1½
!stacked_rnn_cells/lstm_cell/mul_1Mul)stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_1«
 stacked_rnn_cells/lstm_cell/TanhTanh*stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 stacked_rnn_cells/lstm_cell/TanhÏ
!stacked_rnn_cells/lstm_cell/mul_2Mul'stacked_rnn_cells/lstm_cell/Sigmoid:y:0$stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_2Ð
!stacked_rnn_cells/lstm_cell/add_1AddV2%stacked_rnn_cells/lstm_cell/mul_1:z:0%stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/add_1¸
%stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid*stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_2ª
"stacked_rnn_cells/lstm_cell/Tanh_1Tanh%stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/Tanh_1Ó
!stacked_rnn_cells/lstm_cell/mul_3Mul)stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0&stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_3³
-stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape%stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Shape£
-stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Constý
'stacked_rnn_cells/lstm_cell_1/ones_likeFill6stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:06stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/ones_likeÙ
!stacked_rnn_cells/lstm_cell_1/mulMul%stacked_rnn_cells/lstm_cell/mul_3:z:00stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/mulé
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpí
$stacked_rnn_cells/lstm_cell_1/MatMulMatMul%stacked_rnn_cells/lstm_cell_1/mul:z:0;stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/MatMulï
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype027
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÞ
&stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulzeros_2:output:0=stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stacked_rnn_cells/lstm_cell_1/MatMul_1ä
!stacked_rnn_cells/lstm_cell_1/addAddV2.stacked_rnn_cells/lstm_cell_1/MatMul:product:00stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/addç
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpñ
%stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd%stacked_rnn_cells/lstm_cell_1/add:z:0<stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/BiasAdd 
-stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-stacked_rnn_cells/lstm_cell_1/split/split_dim»
#stacked_rnn_cells/lstm_cell_1/splitSplit6stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0.stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#stacked_rnn_cells/lstm_cell_1/splitº
%stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid,stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/Sigmoid¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ã
#stacked_rnn_cells/lstm_cell_1/mul_1Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_1±
"stacked_rnn_cells/lstm_cell_1/TanhTanh,stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell_1/Tanh×
#stacked_rnn_cells/lstm_cell_1/mul_2Mul)stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0&stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_2Ø
#stacked_rnn_cells/lstm_cell_1/add_1AddV2'stacked_rnn_cells/lstm_cell_1/mul_1:z:0'stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/add_1¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2°
$stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh'stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/Tanh_1Û
#stacked_rnn_cells/lstm_cell_1/mul_3Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0(stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61368210*
condR
while_cond_61368209*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1½
IdentityIdentitytranspose_1:y:03^stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2^stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp4^stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp5^stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4^stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp6^stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2h
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2f
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2l
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2n
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¨
¸
rnn_while_body_61367451$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3
rnn_while_placeholder_4
rnn_while_placeholder_5#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0Z
Frnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0:
\
Hrnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0:
V
Grnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0:	\
Hrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0:
^
Jrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0:
X
Irnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0:	
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5
rnn_while_identity_6
rnn_while_identity_7!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorX
Drnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
Z
Frnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
T
Ernn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	Z
Frnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
\
Hrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
V
Grnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	¢<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpË
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2=
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeì
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02/
-rnn/while/TensorArrayV2Read/TensorListGetItemÒ
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape4rnn/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:27
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Shape³
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Const
/rnn/while/stacked_rnn_cells/lstm_cell/ones_likeFill>rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0>rnn/while/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell/ones_like
)rnn/while/stacked_rnn_cells/lstm_cell/mulMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:08rnn/while/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/while/stacked_rnn_cells/lstm_cell/mul
;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpFrnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02=
;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp
,rnn/while/stacked_rnn_cells/lstm_cell/MatMulMatMul-rnn/while/stacked_rnn_cells/lstm_cell/mul:z:0Crnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,rnn/while/stacked_rnn_cells/lstm_cell/MatMul
=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpHrnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02?
=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpý
.rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1MatMulrnn_while_placeholder_2Ernn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1
)rnn/while/stacked_rnn_cells/lstm_cell/addAddV26rnn/while/stacked_rnn_cells/lstm_cell/MatMul:product:08rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)rnn/while/stacked_rnn_cells/lstm_cell/add
<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpGrnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02>
<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
-rnn/while/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd-rnn/while/stacked_rnn_cells/lstm_cell/add:z:0Drnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd°
5rnn/while/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5rnn/while/stacked_rnn_cells/lstm_cell/split/split_dimÛ
+rnn/while/stacked_rnn_cells/lstm_cell/splitSplit>rnn/while/stacked_rnn_cells/lstm_cell/split/split_dim:output:06rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2-
+rnn/while/stacked_rnn_cells/lstm_cell/splitÒ
-rnn/while/stacked_rnn_cells/lstm_cell/SigmoidSigmoid4rnn/while/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell/SigmoidÖ
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid4rnn/while/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1â
+rnn/while/stacked_rnn_cells/lstm_cell/mul_1Mul3rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/mul_1É
*rnn/while/stacked_rnn_cells/lstm_cell/TanhTanh4rnn/while/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*rnn/while/stacked_rnn_cells/lstm_cell/Tanh÷
+rnn/while/stacked_rnn_cells/lstm_cell/mul_2Mul1rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid:y:0.rnn/while/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/mul_2ø
+rnn/while/stacked_rnn_cells/lstm_cell/add_1AddV2/rnn/while/stacked_rnn_cells/lstm_cell/mul_1:z:0/rnn/while/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/add_1Ö
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid4rnn/while/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2È
,rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1Tanh/rnn/while/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1û
+rnn/while/stacked_rnn_cells/lstm_cell/mul_3Mul3rnn/while/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:00rnn/while/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell/mul_3Ñ
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:29
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape·
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Const¥
1rnn/while/stacked_rnn_cells/lstm_cell_1/ones_likeFill@rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0@rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like
+rnn/while/stacked_rnn_cells/lstm_cell_1/mulMul/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0:rnn/while/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell_1/mul
=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpHrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0* 
_output_shapes
:
*
dtype02?
=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
.rnn/while/stacked_rnn_cells/lstm_cell_1/MatMulMatMul/rnn/while/stacked_rnn_cells/lstm_cell_1/mul:z:0Ernn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul
?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpJrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02A
?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp
0rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulrnn_while_placeholder_4Grnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1
+rnn/while/stacked_rnn_cells/lstm_cell_1/addAddV28rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul:product:0:rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+rnn/while/stacked_rnn_cells/lstm_cell_1/add
>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpIrnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02@
>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd/rnn/while/stacked_rnn_cells/lstm_cell_1/add:z:0Frnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd´
7rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dimã
-rnn/while/stacked_rnn_cells/lstm_cell_1/splitSplit@rnn/while/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:08rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/splitØ
/rnn/while/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/rnn/while/stacked_rnn_cells/lstm_cell_1/SigmoidÜ
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1è
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1Mul5rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0rnn_while_placeholder_5*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1Ï
,rnn/while/stacked_rnn_cells/lstm_cell_1/TanhTanh6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,rnn/while/stacked_rnn_cells/lstm_cell_1/Tanhÿ
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2Mul3rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:00rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2
-rnn/while/stacked_rnn_cells/lstm_cell_1/add_1AddV21rnn/while/stacked_rnn_cells/lstm_cell_1/mul_1:z:01rnn/while/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/add_1Ü
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid6rnn/while/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Î
.rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh1rnn/while/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3Mul5rnn/while/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:02rnn/while/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholder1rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0*
_output_shapes
: *
element_dtype020
.rnn/while/TensorArrayV2Write/TensorListSetItemd
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add/yy
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn/while/addh
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn/while/add_1/y
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn/while/add_1ê
rnn/while/IdentityIdentityrnn/while/add_1:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_1ì
rnn/while/Identity_2Identityrnn/while/add:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_2
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn/while/Identity_3
rnn/while/Identity_4Identity/rnn/while/stacked_rnn_cells/lstm_cell/mul_3:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_4
rnn/while/Identity_5Identity/rnn/while/stacked_rnn_cells/lstm_cell/add_1:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_5
rnn/while/Identity_6Identity1rnn/while/stacked_rnn_cells/lstm_cell_1/mul_3:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_6
rnn/while/Identity_7Identity1rnn/while/stacked_rnn_cells/lstm_cell_1/add_1:z:0=^rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<^rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp?^rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp@^rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rnn/while/Identity_7"1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"5
rnn_while_identity_6rnn/while/Identity_6:output:0"5
rnn_while_identity_7rnn/while/Identity_7:output:0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"
Grnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resourceIrnn_while_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource_0"
Hrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceJrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource_0"
Frnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceHrnn_while_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource_0"
Ernn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceGrnn_while_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource_0"
Frnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceHrnn_while_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource_0"
Drnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceFrnn_while_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource_0"¸
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2|
<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp<rnn/while/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2z
;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp;rnn/while/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2~
=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp=rnn/while/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2
>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp>rnn/while/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2~
=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp=rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2
?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp?rnn/while/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
: 
þ	
ã
'__inference_test_layer_call_fn_61367330

inputs
unknown:	S
	unknown_0:

	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:	S
	unknown_7:S
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*+
_read_only_resource_inputs
		*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_test_layer_call_and_return_conditional_losses_613671652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


A__inference_rnn_layer_call_and_return_conditional_losses_61367092

inputsN
:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
P
<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
J
;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	P
<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
R
>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
L
=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	
identity¢2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
+stacked_rnn_cells/lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/ones_like/Shape
+stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+stacked_rnn_cells/lstm_cell/ones_like/Constõ
%stacked_rnn_cells/lstm_cell/ones_likeFill4stacked_rnn_cells/lstm_cell/ones_like/Shape:output:04stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/ones_like
)stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2+
)stacked_rnn_cells/lstm_cell/dropout/Constð
'stacked_rnn_cells/lstm_cell/dropout/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:02stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell/dropout/Mul´
)stacked_rnn_cells/lstm_cell/dropout/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2+
)stacked_rnn_cells/lstm_cell/dropout/Shape¨
@stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform2stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2È±ô2B
@stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform­
2stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>24
2stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y¯
0stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualIstacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0;stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0stacked_rnn_cells/lstm_cell/dropout/GreaterEqualÔ
(stacked_rnn_cells/lstm_cell/dropout/CastCast4stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stacked_rnn_cells/lstm_cell/dropout/Castë
)stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul+stacked_rnn_cells/lstm_cell/dropout/Mul:z:0,stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_1/Constö
)stacked_rnn_cells/lstm_cell/dropout_1/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_1/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_1/Shape­
Bstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2«*2D
Bstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_1/CastCast6stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_1/Castó
+stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_1/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_2/Constö
)stacked_rnn_cells/lstm_cell/dropout_2/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_2/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_2/Shape®
Bstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÄÈÏ2D
Bstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_2/CastCast6stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_2/Castó
+stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_2/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_3/Constö
)stacked_rnn_cells/lstm_cell/dropout_3/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_3/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_3/Shape­
Bstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ý¡%2D
Bstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_3/CastCast6stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_3/Castó
+stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Å
stacked_rnn_cells/lstm_cell/mulMulstrided_slice_2:output:0-stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/mulã
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpå
"stacked_rnn_cells/lstm_cell/MatMulMatMul#stacked_rnn_cells/lstm_cell/mul:z:09stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/MatMulé
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpÖ
$stacked_rnn_cells/lstm_cell/MatMul_1MatMulzeros:output:0;stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell/MatMul_1Ü
stacked_rnn_cells/lstm_cell/addAddV2,stacked_rnn_cells/lstm_cell/MatMul:product:0.stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/addá
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpé
#stacked_rnn_cells/lstm_cell/BiasAddBiasAdd#stacked_rnn_cells/lstm_cell/add:z:0:stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/BiasAdd
+stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+stacked_rnn_cells/lstm_cell/split/split_dim³
!stacked_rnn_cells/lstm_cell/splitSplit4stacked_rnn_cells/lstm_cell/split/split_dim:output:0,stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!stacked_rnn_cells/lstm_cell/split´
#stacked_rnn_cells/lstm_cell/SigmoidSigmoid*stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/Sigmoid¸
%stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid*stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_1½
!stacked_rnn_cells/lstm_cell/mul_1Mul)stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_1«
 stacked_rnn_cells/lstm_cell/TanhTanh*stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 stacked_rnn_cells/lstm_cell/TanhÏ
!stacked_rnn_cells/lstm_cell/mul_2Mul'stacked_rnn_cells/lstm_cell/Sigmoid:y:0$stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_2Ð
!stacked_rnn_cells/lstm_cell/add_1AddV2%stacked_rnn_cells/lstm_cell/mul_1:z:0%stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/add_1¸
%stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid*stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_2ª
"stacked_rnn_cells/lstm_cell/Tanh_1Tanh%stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/Tanh_1Ó
!stacked_rnn_cells/lstm_cell/mul_3Mul)stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0&stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_3³
-stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape%stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Shape£
-stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Constý
'stacked_rnn_cells/lstm_cell_1/ones_likeFill6stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:06stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/ones_like
+stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell_1/dropout/Constø
)stacked_rnn_cells/lstm_cell_1/dropout/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:04stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell_1/dropout/Mulº
+stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell_1/dropout/Shape®
Bstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÆÔ2D
Bstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell_1/dropout/CastCast6stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell_1/dropout/Castó
+stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul-stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:0.stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_1/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Shape´
Dstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2þÔ2F
Dstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_1/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_1/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_2/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Shape´
Dstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2µûò2F
Dstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_2/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_2/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_3/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Shape´
Dstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¼£ª2F
Dstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_3/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_3/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Ø
!stacked_rnn_cells/lstm_cell_1/mulMul%stacked_rnn_cells/lstm_cell/mul_3:z:0/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/mulé
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpí
$stacked_rnn_cells/lstm_cell_1/MatMulMatMul%stacked_rnn_cells/lstm_cell_1/mul:z:0;stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/MatMulï
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype027
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÞ
&stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulzeros_2:output:0=stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stacked_rnn_cells/lstm_cell_1/MatMul_1ä
!stacked_rnn_cells/lstm_cell_1/addAddV2.stacked_rnn_cells/lstm_cell_1/MatMul:product:00stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/addç
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpñ
%stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd%stacked_rnn_cells/lstm_cell_1/add:z:0<stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/BiasAdd 
-stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-stacked_rnn_cells/lstm_cell_1/split/split_dim»
#stacked_rnn_cells/lstm_cell_1/splitSplit6stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0.stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#stacked_rnn_cells/lstm_cell_1/splitº
%stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid,stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/Sigmoid¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ã
#stacked_rnn_cells/lstm_cell_1/mul_1Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_1±
"stacked_rnn_cells/lstm_cell_1/TanhTanh,stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell_1/Tanh×
#stacked_rnn_cells/lstm_cell_1/mul_2Mul)stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0&stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_2Ø
#stacked_rnn_cells/lstm_cell_1/add_1AddV2'stacked_rnn_cells/lstm_cell_1/mul_1:z:0'stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/add_1¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2°
$stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh'stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/Tanh_1Û
#stacked_rnn_cells/lstm_cell_1/mul_3Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0(stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61366899*
condR
while_cond_61366898*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1½
IdentityIdentitytranspose_1:y:03^stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2^stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp4^stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp5^stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4^stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp6^stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2h
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2f
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2l
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2n
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


A__inference_rnn_layer_call_and_return_conditional_losses_61368711
inputs_0N
:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
P
<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
J
;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	P
<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
R
>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
L
=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	
identity¢2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
+stacked_rnn_cells/lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/ones_like/Shape
+stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+stacked_rnn_cells/lstm_cell/ones_like/Constõ
%stacked_rnn_cells/lstm_cell/ones_likeFill4stacked_rnn_cells/lstm_cell/ones_like/Shape:output:04stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/ones_like
)stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2+
)stacked_rnn_cells/lstm_cell/dropout/Constð
'stacked_rnn_cells/lstm_cell/dropout/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:02stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell/dropout/Mul´
)stacked_rnn_cells/lstm_cell/dropout/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2+
)stacked_rnn_cells/lstm_cell/dropout/Shape¨
@stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform2stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2àÒÙ2B
@stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform­
2stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>24
2stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y¯
0stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualIstacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0;stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0stacked_rnn_cells/lstm_cell/dropout/GreaterEqualÔ
(stacked_rnn_cells/lstm_cell/dropout/CastCast4stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stacked_rnn_cells/lstm_cell/dropout/Castë
)stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul+stacked_rnn_cells/lstm_cell/dropout/Mul:z:0,stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_1/Constö
)stacked_rnn_cells/lstm_cell/dropout_1/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_1/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_1/Shape®
Bstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ã¶2D
Bstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_1/CastCast6stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_1/Castó
+stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_1/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_2/Constö
)stacked_rnn_cells/lstm_cell/dropout_2/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_2/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_2/Shape®
Bstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2®í2D
Bstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_2/CastCast6stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_2/Castó
+stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_2/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_3/Constö
)stacked_rnn_cells/lstm_cell/dropout_3/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_3/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_3/Shape®
Bstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ï°2D
Bstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_3/CastCast6stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_3/Castó
+stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Å
stacked_rnn_cells/lstm_cell/mulMulstrided_slice_2:output:0-stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/mulã
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpå
"stacked_rnn_cells/lstm_cell/MatMulMatMul#stacked_rnn_cells/lstm_cell/mul:z:09stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/MatMulé
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpÖ
$stacked_rnn_cells/lstm_cell/MatMul_1MatMulzeros:output:0;stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell/MatMul_1Ü
stacked_rnn_cells/lstm_cell/addAddV2,stacked_rnn_cells/lstm_cell/MatMul:product:0.stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/addá
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpé
#stacked_rnn_cells/lstm_cell/BiasAddBiasAdd#stacked_rnn_cells/lstm_cell/add:z:0:stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/BiasAdd
+stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+stacked_rnn_cells/lstm_cell/split/split_dim³
!stacked_rnn_cells/lstm_cell/splitSplit4stacked_rnn_cells/lstm_cell/split/split_dim:output:0,stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!stacked_rnn_cells/lstm_cell/split´
#stacked_rnn_cells/lstm_cell/SigmoidSigmoid*stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/Sigmoid¸
%stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid*stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_1½
!stacked_rnn_cells/lstm_cell/mul_1Mul)stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_1«
 stacked_rnn_cells/lstm_cell/TanhTanh*stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 stacked_rnn_cells/lstm_cell/TanhÏ
!stacked_rnn_cells/lstm_cell/mul_2Mul'stacked_rnn_cells/lstm_cell/Sigmoid:y:0$stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_2Ð
!stacked_rnn_cells/lstm_cell/add_1AddV2%stacked_rnn_cells/lstm_cell/mul_1:z:0%stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/add_1¸
%stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid*stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_2ª
"stacked_rnn_cells/lstm_cell/Tanh_1Tanh%stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/Tanh_1Ó
!stacked_rnn_cells/lstm_cell/mul_3Mul)stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0&stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_3³
-stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape%stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Shape£
-stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Constý
'stacked_rnn_cells/lstm_cell_1/ones_likeFill6stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:06stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/ones_like
+stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell_1/dropout/Constø
)stacked_rnn_cells/lstm_cell_1/dropout/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:04stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell_1/dropout/Mulº
+stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell_1/dropout/Shape®
Bstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ìÓ2D
Bstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell_1/dropout/CastCast6stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell_1/dropout/Castó
+stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul-stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:0.stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_1/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Shape´
Dstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Áü2F
Dstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_1/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_1/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_2/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Shape³
Dstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¤Õ,2F
Dstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_2/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_2/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_3/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Shape³
Dstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2àöH2F
Dstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_3/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_3/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Ø
!stacked_rnn_cells/lstm_cell_1/mulMul%stacked_rnn_cells/lstm_cell/mul_3:z:0/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/mulé
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpí
$stacked_rnn_cells/lstm_cell_1/MatMulMatMul%stacked_rnn_cells/lstm_cell_1/mul:z:0;stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/MatMulï
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype027
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÞ
&stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulzeros_2:output:0=stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stacked_rnn_cells/lstm_cell_1/MatMul_1ä
!stacked_rnn_cells/lstm_cell_1/addAddV2.stacked_rnn_cells/lstm_cell_1/MatMul:product:00stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/addç
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpñ
%stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd%stacked_rnn_cells/lstm_cell_1/add:z:0<stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/BiasAdd 
-stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-stacked_rnn_cells/lstm_cell_1/split/split_dim»
#stacked_rnn_cells/lstm_cell_1/splitSplit6stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0.stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#stacked_rnn_cells/lstm_cell_1/splitº
%stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid,stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/Sigmoid¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ã
#stacked_rnn_cells/lstm_cell_1/mul_1Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_1±
"stacked_rnn_cells/lstm_cell_1/TanhTanh,stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell_1/Tanh×
#stacked_rnn_cells/lstm_cell_1/mul_2Mul)stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0&stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_2Ø
#stacked_rnn_cells/lstm_cell_1/add_1AddV2'stacked_rnn_cells/lstm_cell_1/mul_1:z:0'stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/add_1¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2°
$stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh'stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/Tanh_1Û
#stacked_rnn_cells/lstm_cell_1/mul_3Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0(stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61368518*
condR
while_cond_61368517*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1½
IdentityIdentitytranspose_1:y:03^stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2^stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp4^stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp5^stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4^stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp6^stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2h
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2f
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2l
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2n
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ó

(__inference_dense_layer_call_fn_61369336

inputs
unknown:	S
	unknown_0:S
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_613666632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï©
¼
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61369624

inputs

states_0_0

states_0_1

states_1_0

states_1_1<
(lstm_cell_matmul_readvariableop_resource:
>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	>
*lstm_cell_1_matmul_readvariableop_resource:
@
,lstm_cell_1_matmul_1_readvariableop_resource:
:
+lstm_cell_1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3

identity_4¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢"lstm_cell_1/BiasAdd/ReadVariableOp¢!lstm_cell_1/MatMul/ReadVariableOp¢#lstm_cell_1/MatMul_1/ReadVariableOpl
lstm_cell/ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout/Const¨
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeò
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2èË§20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 lstm_cell/dropout/GreaterEqual/yç
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout/Cast£
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout_1/Const®
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeø
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ºÇØ22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell/dropout_1/GreaterEqual/yï
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell/dropout_1/GreaterEqual¤
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_1/Cast«
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout_2/Const®
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeø
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÏÃ22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell/dropout_2/GreaterEqual/yï
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell/dropout_2/GreaterEqual¤
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_2/Cast«
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout_3/Const®
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shape÷
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2­422
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell/dropout_3/GreaterEqual/yï
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell/dropout_3/GreaterEqual¤
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_3/Cast«
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_3/Mul_1}
lstm_cell/mulMulinputslstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMullstm_cell/mul:z:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul³
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMul
states_0_0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add«
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp¡
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimë
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell/split~
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_1
lstm_cell/mul_1Mullstm_cell/Sigmoid_1:y:0
states_0_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_1u
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh
lstm_cell/mul_2Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2
lstm_cell/add_1AddV2lstm_cell/mul_1:z:0lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh_1
lstm_cell/mul_3Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_3}
lstm_cell_1/ones_like/ShapeShapelstm_cell/mul_3:z:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout/Const°
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shape÷
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2°ì!22
0lstm_cell_1/dropout/random_uniform/RandomUniform
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell_1/dropout/GreaterEqual/yï
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell_1/dropout/GreaterEqual¤
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Cast«
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout_1/Const¶
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shapeý
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ç¾s24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_1/dropout_1/GreaterEqual/y÷
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_1/GreaterEqualª
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Cast³
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout_2/Const¶
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shapeþ
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÌËð24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_1/dropout_2/GreaterEqual/y÷
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_2/GreaterEqualª
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Cast³
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout_3/Const¶
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shapeþ
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§Ó°24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_1/dropout_3/GreaterEqual/y÷
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_3/GreaterEqualª
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Cast³
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul_1
lstm_cell_1/mulMullstm_cell/mul_3:z:0lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul³
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell_1/MatMul/ReadVariableOp¥
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul¹
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_cell_1/MatMul_1/ReadVariableOp¢
lstm_cell_1/MatMul_1MatMul
states_1_0+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add±
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/BiasAdd/ReadVariableOp©
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dimó
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_1/split
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid_1:y:0
states_1_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1{
lstm_cell_1/TanhTanhlstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/add_1AddV2lstm_cell_1/mul_1:z:0lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_3Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3Â
IdentityIdentitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÄ

Identity_1Identitylstm_cell/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ä

Identity_2Identitylstm_cell/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2Æ

Identity_3Identitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3Æ

Identity_4Identitylstm_cell_1/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/0/1:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/0:TP
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
states/1/1
Û
î
B__inference_test_layer_call_and_return_conditional_losses_61367259
input_1%
embedding_61367237:	S 
rnn_61367240:
 
rnn_61367242:

rnn_61367244:	 
rnn_61367246:
 
rnn_61367248:

rnn_61367250:	!
dense_61367253:	S
dense_61367255:S
identity¢dense/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢rnn/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_61367237*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_613663722#
!embedding/StatefulPartitionedCallö
rnn/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0rnn_61367240rnn_61367242rnn_61367244rnn_61367246rnn_61367248rnn_61367250*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613670922
rnn/StatefulPartitionedCall¹
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_61367253dense_61367255*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_613666632
dense/StatefulPartitionedCallé
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ì©
²
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61365783

inputs

states
states_1
states_2
states_3<
(lstm_cell_matmul_readvariableop_resource:
>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	>
*lstm_cell_1_matmul_readvariableop_resource:
@
,lstm_cell_1_matmul_1_readvariableop_resource:
:
+lstm_cell_1_biasadd_readvariableop_resource:	
identity

identity_1

identity_2

identity_3

identity_4¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢"lstm_cell_1/BiasAdd/ReadVariableOp¢!lstm_cell_1/MatMul/ReadVariableOp¢#lstm_cell_1/MatMul_1/ReadVariableOpl
lstm_cell/ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_cell/ones_like/Shape{
lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell/ones_like/Const­
lstm_cell/ones_likeFill"lstm_cell/ones_like/Shape:output:0"lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/ones_likew
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout/Const¨
lstm_cell/dropout/MulMullstm_cell/ones_like:output:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout/Mul~
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout/Shapeò
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2°àÀ20
.lstm_cell/dropout/random_uniform/RandomUniform
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 lstm_cell/dropout/GreaterEqual/yç
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_cell/dropout/GreaterEqual
lstm_cell/dropout/CastCast"lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout/Cast£
lstm_cell/dropout/Mul_1Mullstm_cell/dropout/Mul:z:0lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout/Mul_1{
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout_1/Const®
lstm_cell/dropout_1/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_1/Mul
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_1/Shapeø
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2õÕü22
0lstm_cell/dropout_1/random_uniform/RandomUniform
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell/dropout_1/GreaterEqual/yï
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell/dropout_1/GreaterEqual¤
lstm_cell/dropout_1/CastCast$lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_1/Cast«
lstm_cell/dropout_1/Mul_1Mullstm_cell/dropout_1/Mul:z:0lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_1/Mul_1{
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout_2/Const®
lstm_cell/dropout_2/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_2/Mul
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_2/Shapeø
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Õí22
0lstm_cell/dropout_2/random_uniform/RandomUniform
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell/dropout_2/GreaterEqual/yï
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell/dropout_2/GreaterEqual¤
lstm_cell/dropout_2/CastCast$lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_2/Cast«
lstm_cell/dropout_2/Mul_1Mullstm_cell/dropout_2/Mul:z:0lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_2/Mul_1{
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell/dropout_3/Const®
lstm_cell/dropout_3/MulMullstm_cell/ones_like:output:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_3/Mul
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell/dropout_3/Shapeø
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¢þ22
0lstm_cell/dropout_3/random_uniform/RandomUniform
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell/dropout_3/GreaterEqual/yï
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell/dropout_3/GreaterEqual¤
lstm_cell/dropout_3/CastCast$lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_3/Cast«
lstm_cell/dropout_3/Mul_1Mullstm_cell/dropout_3/Mul:z:0lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/dropout_3/Mul_1}
lstm_cell/mulMulinputslstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul­
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
lstm_cell/MatMul/ReadVariableOp
lstm_cell/MatMulMatMullstm_cell/mul:z:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul³
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp
lstm_cell/MatMul_1MatMulstates)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/MatMul_1
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add«
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp¡
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/BiasAddx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimë
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell/split~
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_1
lstm_cell/mul_1Mullstm_cell/Sigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_1u
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh
lstm_cell/mul_2Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_2
lstm_cell/add_1AddV2lstm_cell/mul_1:z:0lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/add_1
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Sigmoid_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/Tanh_1
lstm_cell/mul_3Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell/mul_3}
lstm_cell_1/ones_like/ShapeShapelstm_cell/mul_3:z:0*
T0*
_output_shapes
:2
lstm_cell_1/ones_like/Shape
lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lstm_cell_1/ones_like/Constµ
lstm_cell_1/ones_likeFill$lstm_cell_1/ones_like/Shape:output:0$lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/ones_like{
lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout/Const°
lstm_cell_1/dropout/MulMullstm_cell_1/ones_like:output:0"lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul
lstm_cell_1/dropout/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout/Shapeø
0lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Ö 22
0lstm_cell_1/dropout/random_uniform/RandomUniform
"lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"lstm_cell_1/dropout/GreaterEqual/yï
 lstm_cell_1/dropout/GreaterEqualGreaterEqual9lstm_cell_1/dropout/random_uniform/RandomUniform:output:0+lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_cell_1/dropout/GreaterEqual¤
lstm_cell_1/dropout/CastCast$lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Cast«
lstm_cell_1/dropout/Mul_1Mullstm_cell_1/dropout/Mul:z:0lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout/Mul_1
lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout_1/Const¶
lstm_cell_1/dropout_1/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul
lstm_cell_1/dropout_1/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_1/Shapeþ
2lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Á÷24
2lstm_cell_1/dropout_1/random_uniform/RandomUniform
$lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_1/dropout_1/GreaterEqual/y÷
"lstm_cell_1/dropout_1/GreaterEqualGreaterEqual;lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_1/GreaterEqualª
lstm_cell_1/dropout_1/CastCast&lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Cast³
lstm_cell_1/dropout_1/Mul_1Mullstm_cell_1/dropout_1/Mul:z:0lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_1/Mul_1
lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout_2/Const¶
lstm_cell_1/dropout_2/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul
lstm_cell_1/dropout_2/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_2/Shapeþ
2lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2 ËÈ24
2lstm_cell_1/dropout_2/random_uniform/RandomUniform
$lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_1/dropout_2/GreaterEqual/y÷
"lstm_cell_1/dropout_2/GreaterEqualGreaterEqual;lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_2/GreaterEqualª
lstm_cell_1/dropout_2/CastCast&lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Cast³
lstm_cell_1/dropout_2/Mul_1Mullstm_cell_1/dropout_2/Mul:z:0lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_2/Mul_1
lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
lstm_cell_1/dropout_3/Const¶
lstm_cell_1/dropout_3/MulMullstm_cell_1/ones_like:output:0$lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul
lstm_cell_1/dropout_3/ShapeShapelstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_1/dropout_3/Shapeý
2lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2üÖa24
2lstm_cell_1/dropout_3/random_uniform/RandomUniform
$lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$lstm_cell_1/dropout_3/GreaterEqual/y÷
"lstm_cell_1/dropout_3/GreaterEqualGreaterEqual;lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0-lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_cell_1/dropout_3/GreaterEqualª
lstm_cell_1/dropout_3/CastCast&lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Cast³
lstm_cell_1/dropout_3/Mul_1Mullstm_cell_1/dropout_3/Mul:z:0lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/dropout_3/Mul_1
lstm_cell_1/mulMullstm_cell/mul_3:z:0lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul³
!lstm_cell_1/MatMul/ReadVariableOpReadVariableOp*lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!lstm_cell_1/MatMul/ReadVariableOp¥
lstm_cell_1/MatMulMatMullstm_cell_1/mul:z:0)lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul¹
#lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02%
#lstm_cell_1/MatMul_1/ReadVariableOp 
lstm_cell_1/MatMul_1MatMulstates_2+lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/MatMul_1
lstm_cell_1/addAddV2lstm_cell_1/MatMul:product:0lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add±
"lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_1/BiasAdd/ReadVariableOp©
lstm_cell_1/BiasAddBiasAddlstm_cell_1/add:z:0*lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/BiasAdd|
lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_1/split/split_dimó
lstm_cell_1/splitSplit$lstm_cell_1/split/split_dim:output:0lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_1/split
lstm_cell_1/SigmoidSigmoidlstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid
lstm_cell_1/Sigmoid_1Sigmoidlstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_1
lstm_cell_1/mul_1Mullstm_cell_1/Sigmoid_1:y:0states_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_1{
lstm_cell_1/TanhTanhlstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh
lstm_cell_1/mul_2Mullstm_cell_1/Sigmoid:y:0lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_2
lstm_cell_1/add_1AddV2lstm_cell_1/mul_1:z:0lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/add_1
lstm_cell_1/Sigmoid_2Sigmoidlstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Sigmoid_2z
lstm_cell_1/Tanh_1Tanhlstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/Tanh_1
lstm_cell_1/mul_3Mullstm_cell_1/Sigmoid_2:y:0lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_1/mul_3Â
IdentityIdentitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÄ

Identity_1Identitylstm_cell/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1Ä

Identity_2Identitylstm_cell/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2Æ

Identity_3Identitylstm_cell_1/mul_3:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_3Æ

Identity_4Identitylstm_cell_1/add_1:z:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp#^lstm_cell_1/BiasAdd/ReadVariableOp"^lstm_cell_1/MatMul/ReadVariableOp$^lstm_cell_1/MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2H
"lstm_cell_1/BiasAdd/ReadVariableOp"lstm_cell_1/BiasAdd/ReadVariableOp2F
!lstm_cell_1/MatMul/ReadVariableOp!lstm_cell_1/MatMul/ReadVariableOp2J
#lstm_cell_1/MatMul_1/ReadVariableOp#lstm_cell_1/MatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
	

&__inference_rnn_layer_call_fn_61368095

inputs
unknown:

	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:	
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613670922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Åñ


#__inference__wrapped_model_61365361
input_1;
(test_embedding_embedding_lookup_61365089:	SW
Ctest_rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
Y
Etest_rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
S
Dtest_rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	Y
Etest_rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
[
Gtest_rnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
U
Ftest_rnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	?
,test_dense_tensordot_readvariableop_resource:	S8
*test_dense_biasadd_readvariableop_resource:S
identity¢!test/dense/BiasAdd/ReadVariableOp¢#test/dense/Tensordot/ReadVariableOp¢test/embedding/embedding_lookup¢;test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢:test/rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢<test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢=test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢<test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢>test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢test/rnn/while
test/embedding/CastCastinput_1*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
test/embedding/CastÖ
test/embedding/embedding_lookupResourceGather(test_embedding_embedding_lookup_61365089test/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@test/embedding/embedding_lookup/61365089*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02!
test/embedding/embedding_lookupµ
(test/embedding/embedding_lookup/IdentityIdentity(test/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@test/embedding/embedding_lookup/61365089*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(test/embedding/embedding_lookup/Identity×
*test/embedding/embedding_lookup/Identity_1Identity1test/embedding/embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*test/embedding/embedding_lookup/Identity_1
test/rnn/ShapeShape3test/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
test/rnn/Shape
test/rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
test/rnn/strided_slice/stack
test/rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
test/rnn/strided_slice/stack_1
test/rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
test/rnn/strided_slice/stack_2
test/rnn/strided_sliceStridedSlicetest/rnn/Shape:output:0%test/rnn/strided_slice/stack:output:0'test/rnn/strided_slice/stack_1:output:0'test/rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
test/rnn/strided_sliceo
test/rnn/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros/mul/y
test/rnn/zeros/mulMultest/rnn/strided_slice:output:0test/rnn/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros/mulq
test/rnn/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
test/rnn/zeros/Less/y
test/rnn/zeros/LessLesstest/rnn/zeros/mul:z:0test/rnn/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros/Lessu
test/rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros/packed/1§
test/rnn/zeros/packedPacktest/rnn/strided_slice:output:0 test/rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
test/rnn/zeros/packedq
test/rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
test/rnn/zeros/Const
test/rnn/zerosFilltest/rnn/zeros/packed:output:0test/rnn/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/zeross
test/rnn/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros_1/mul/y
test/rnn/zeros_1/mulMultest/rnn/strided_slice:output:0test/rnn/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros_1/mulu
test/rnn/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
test/rnn/zeros_1/Less/y
test/rnn/zeros_1/LessLesstest/rnn/zeros_1/mul:z:0 test/rnn/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros_1/Lessy
test/rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros_1/packed/1­
test/rnn/zeros_1/packedPacktest/rnn/strided_slice:output:0"test/rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
test/rnn/zeros_1/packedu
test/rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
test/rnn/zeros_1/Const¢
test/rnn/zeros_1Fill test/rnn/zeros_1/packed:output:0test/rnn/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/zeros_1s
test/rnn/zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros_2/mul/y
test/rnn/zeros_2/mulMultest/rnn/strided_slice:output:0test/rnn/zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros_2/mulu
test/rnn/zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
test/rnn/zeros_2/Less/y
test/rnn/zeros_2/LessLesstest/rnn/zeros_2/mul:z:0 test/rnn/zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros_2/Lessy
test/rnn/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros_2/packed/1­
test/rnn/zeros_2/packedPacktest/rnn/strided_slice:output:0"test/rnn/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
test/rnn/zeros_2/packedu
test/rnn/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
test/rnn/zeros_2/Const¢
test/rnn/zeros_2Fill test/rnn/zeros_2/packed:output:0test/rnn/zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/zeros_2s
test/rnn/zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros_3/mul/y
test/rnn/zeros_3/mulMultest/rnn/strided_slice:output:0test/rnn/zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros_3/mulu
test/rnn/zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
test/rnn/zeros_3/Less/y
test/rnn/zeros_3/LessLesstest/rnn/zeros_3/mul:z:0 test/rnn/zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
test/rnn/zeros_3/Lessy
test/rnn/zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
test/rnn/zeros_3/packed/1­
test/rnn/zeros_3/packedPacktest/rnn/strided_slice:output:0"test/rnn/zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
test/rnn/zeros_3/packedu
test/rnn/zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
test/rnn/zeros_3/Const¢
test/rnn/zeros_3Fill test/rnn/zeros_3/packed:output:0test/rnn/zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
test/rnn/zeros_3
test/rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
test/rnn/transpose/permÌ
test/rnn/transpose	Transpose3test/embedding/embedding_lookup/Identity_1:output:0 test/rnn/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
test/rnn/transposej
test/rnn/Shape_1Shapetest/rnn/transpose:y:0*
T0*
_output_shapes
:2
test/rnn/Shape_1
test/rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
test/rnn/strided_slice_1/stack
 test/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 test/rnn/strided_slice_1/stack_1
 test/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 test/rnn/strided_slice_1/stack_2¤
test/rnn/strided_slice_1StridedSlicetest/rnn/Shape_1:output:0'test/rnn/strided_slice_1/stack:output:0)test/rnn/strided_slice_1/stack_1:output:0)test/rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
test/rnn/strided_slice_1
$test/rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$test/rnn/TensorArrayV2/element_shapeÖ
test/rnn/TensorArrayV2TensorListReserve-test/rnn/TensorArrayV2/element_shape:output:0!test/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
test/rnn/TensorArrayV2Ñ
>test/rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2@
>test/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape
0test/rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensortest/rnn/transpose:y:0Gtest/rnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0test/rnn/TensorArrayUnstack/TensorListFromTensor
test/rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
test/rnn/strided_slice_2/stack
 test/rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 test/rnn/strided_slice_2/stack_1
 test/rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 test/rnn/strided_slice_2/stack_2³
test/rnn/strided_slice_2StridedSlicetest/rnn/transpose:y:0'test/rnn/strided_slice_2/stack:output:0)test/rnn/strided_slice_2/stack_1:output:0)test/rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
test/rnn/strided_slice_2½
4test/rnn/stacked_rnn_cells/lstm_cell/ones_like/ShapeShape!test/rnn/strided_slice_2:output:0*
T0*
_output_shapes
:26
4test/rnn/stacked_rnn_cells/lstm_cell/ones_like/Shape±
4test/rnn/stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4test/rnn/stacked_rnn_cells/lstm_cell/ones_like/Const
.test/rnn/stacked_rnn_cells/lstm_cell/ones_likeFill=test/rnn/stacked_rnn_cells/lstm_cell/ones_like/Shape:output:0=test/rnn/stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.test/rnn/stacked_rnn_cells/lstm_cell/ones_likeê
(test/rnn/stacked_rnn_cells/lstm_cell/mulMul!test/rnn/strided_slice_2:output:07test/rnn/stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(test/rnn/stacked_rnn_cells/lstm_cell/mulþ
:test/rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOpCtest_rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:test/rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp
+test/rnn/stacked_rnn_cells/lstm_cell/MatMulMatMul,test/rnn/stacked_rnn_cells/lstm_cell/mul:z:0Btest/rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+test/rnn/stacked_rnn_cells/lstm_cell/MatMul
<test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEtest_rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpú
-test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1MatMultest/rnn/zeros:output:0Dtest/rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1
(test/rnn/stacked_rnn_cells/lstm_cell/addAddV25test/rnn/stacked_rnn_cells/lstm_cell/MatMul:product:07test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(test/rnn/stacked_rnn_cells/lstm_cell/addü
;test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDtest_rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp
,test/rnn/stacked_rnn_cells/lstm_cell/BiasAddBiasAdd,test/rnn/stacked_rnn_cells/lstm_cell/add:z:0Ctest/rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd®
4test/rnn/stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4test/rnn/stacked_rnn_cells/lstm_cell/split/split_dim×
*test/rnn/stacked_rnn_cells/lstm_cell/splitSplit=test/rnn/stacked_rnn_cells/lstm_cell/split/split_dim:output:05test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2,
*test/rnn/stacked_rnn_cells/lstm_cell/splitÏ
,test/rnn/stacked_rnn_cells/lstm_cell/SigmoidSigmoid3test/rnn/stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,test/rnn/stacked_rnn_cells/lstm_cell/SigmoidÓ
.test/rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid3test/rnn/stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.test/rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1á
*test/rnn/stacked_rnn_cells/lstm_cell/mul_1Mul2test/rnn/stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0test/rnn/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*test/rnn/stacked_rnn_cells/lstm_cell/mul_1Æ
)test/rnn/stacked_rnn_cells/lstm_cell/TanhTanh3test/rnn/stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)test/rnn/stacked_rnn_cells/lstm_cell/Tanhó
*test/rnn/stacked_rnn_cells/lstm_cell/mul_2Mul0test/rnn/stacked_rnn_cells/lstm_cell/Sigmoid:y:0-test/rnn/stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*test/rnn/stacked_rnn_cells/lstm_cell/mul_2ô
*test/rnn/stacked_rnn_cells/lstm_cell/add_1AddV2.test/rnn/stacked_rnn_cells/lstm_cell/mul_1:z:0.test/rnn/stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*test/rnn/stacked_rnn_cells/lstm_cell/add_1Ó
.test/rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid3test/rnn/stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.test/rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2Å
+test/rnn/stacked_rnn_cells/lstm_cell/Tanh_1Tanh.test/rnn/stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+test/rnn/stacked_rnn_cells/lstm_cell/Tanh_1÷
*test/rnn/stacked_rnn_cells/lstm_cell/mul_3Mul2test/rnn/stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0/test/rnn/stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*test/rnn/stacked_rnn_cells/lstm_cell/mul_3Î
6test/rnn/stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape.test/rnn/stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:28
6test/rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Shapeµ
6test/rnn/stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6test/rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Const¡
0test/rnn/stacked_rnn_cells/lstm_cell_1/ones_likeFill?test/rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:0?test/rnn/stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/stacked_rnn_cells/lstm_cell_1/ones_likeý
*test/rnn/stacked_rnn_cells/lstm_cell_1/mulMul.test/rnn/stacked_rnn_cells/lstm_cell/mul_3:z:09test/rnn/stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*test/rnn/stacked_rnn_cells/lstm_cell_1/mul
<test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOpEtest_rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp
-test/rnn/stacked_rnn_cells/lstm_cell_1/MatMulMatMul.test/rnn/stacked_rnn_cells/lstm_cell_1/mul:z:0Dtest/rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul
>test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOpGtest_rnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02@
>test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp
/test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1MatMultest/rnn/zeros_2:output:0Ftest/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1
*test/rnn/stacked_rnn_cells/lstm_cell_1/addAddV27test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul:product:09test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*test/rnn/stacked_rnn_cells/lstm_cell_1/add
=test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOpFtest_rnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02?
=test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp
.test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd.test/rnn/stacked_rnn_cells/lstm_cell_1/add:z:0Etest/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd²
6test/rnn/stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6test/rnn/stacked_rnn_cells/lstm_cell_1/split/split_dimß
,test/rnn/stacked_rnn_cells/lstm_cell_1/splitSplit?test/rnn/stacked_rnn_cells/lstm_cell_1/split/split_dim:output:07test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2.
,test/rnn/stacked_rnn_cells/lstm_cell_1/splitÕ
.test/rnn/stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid5test/rnn/stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.test/rnn/stacked_rnn_cells/lstm_cell_1/SigmoidÙ
0test/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid5test/rnn/stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1ç
,test/rnn/stacked_rnn_cells/lstm_cell_1/mul_1Mul4test/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0test/rnn/zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,test/rnn/stacked_rnn_cells/lstm_cell_1/mul_1Ì
+test/rnn/stacked_rnn_cells/lstm_cell_1/TanhTanh5test/rnn/stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+test/rnn/stacked_rnn_cells/lstm_cell_1/Tanhû
,test/rnn/stacked_rnn_cells/lstm_cell_1/mul_2Mul2test/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0/test/rnn/stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,test/rnn/stacked_rnn_cells/lstm_cell_1/mul_2ü
,test/rnn/stacked_rnn_cells/lstm_cell_1/add_1AddV20test/rnn/stacked_rnn_cells/lstm_cell_1/mul_1:z:00test/rnn/stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,test/rnn/stacked_rnn_cells/lstm_cell_1/add_1Ù
0test/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid5test/rnn/stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0test/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2Ë
-test/rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh0test/rnn/stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-test/rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1ÿ
,test/rnn/stacked_rnn_cells/lstm_cell_1/mul_3Mul4test/rnn/stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:01test/rnn/stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,test/rnn/stacked_rnn_cells/lstm_cell_1/mul_3¡
&test/rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2(
&test/rnn/TensorArrayV2_1/element_shapeÜ
test/rnn/TensorArrayV2_1TensorListReserve/test/rnn/TensorArrayV2_1/element_shape:output:0!test/rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
test/rnn/TensorArrayV2_1`
test/rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
test/rnn/time
!test/rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!test/rnn/while/maximum_iterations|
test/rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
test/rnn/while/loop_counter

test/rnn/whileWhile$test/rnn/while/loop_counter:output:0*test/rnn/while/maximum_iterations:output:0test/rnn/time:output:0!test/rnn/TensorArrayV2_1:handle:0test/rnn/zeros:output:0test/rnn/zeros_1:output:0test/rnn/zeros_2:output:0test/rnn/zeros_3:output:0!test/rnn/strided_slice_1:output:0@test/rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ctest_rnn_stacked_rnn_cells_lstm_cell_matmul_readvariableop_resourceEtest_rnn_stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resourceDtest_rnn_stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resourceEtest_rnn_stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resourceGtest_rnn_stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resourceFtest_rnn_stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*(
body R
test_rnn_while_body_61365206*(
cond R
test_rnn_while_cond_61365205*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
test/rnn/whileÇ
9test/rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2;
9test/rnn/TensorArrayV2Stack/TensorListStack/element_shape
+test/rnn/TensorArrayV2Stack/TensorListStackTensorListStacktest/rnn/while:output:3Btest/rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02-
+test/rnn/TensorArrayV2Stack/TensorListStack
test/rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
test/rnn/strided_slice_3/stack
 test/rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 test/rnn/strided_slice_3/stack_1
 test/rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 test/rnn/strided_slice_3/stack_2Ñ
test/rnn/strided_slice_3StridedSlice4test/rnn/TensorArrayV2Stack/TensorListStack:tensor:0'test/rnn/strided_slice_3/stack:output:0)test/rnn/strided_slice_3/stack_1:output:0)test/rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
test/rnn/strided_slice_3
test/rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
test/rnn/transpose_1/permÓ
test/rnn/transpose_1	Transpose4test/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"test/rnn/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
test/rnn/transpose_1¸
#test/dense/Tensordot/ReadVariableOpReadVariableOp,test_dense_tensordot_readvariableop_resource*
_output_shapes
:	S*
dtype02%
#test/dense/Tensordot/ReadVariableOp
test/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
test/dense/Tensordot/axes
test/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
test/dense/Tensordot/free
test/dense/Tensordot/ShapeShapetest/rnn/transpose_1:y:0*
T0*
_output_shapes
:2
test/dense/Tensordot/Shape
"test/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"test/dense/Tensordot/GatherV2/axis
test/dense/Tensordot/GatherV2GatherV2#test/dense/Tensordot/Shape:output:0"test/dense/Tensordot/free:output:0+test/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
test/dense/Tensordot/GatherV2
$test/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$test/dense/Tensordot/GatherV2_1/axis
test/dense/Tensordot/GatherV2_1GatherV2#test/dense/Tensordot/Shape:output:0"test/dense/Tensordot/axes:output:0-test/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2!
test/dense/Tensordot/GatherV2_1
test/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
test/dense/Tensordot/Const¬
test/dense/Tensordot/ProdProd&test/dense/Tensordot/GatherV2:output:0#test/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
test/dense/Tensordot/Prod
test/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
test/dense/Tensordot/Const_1´
test/dense/Tensordot/Prod_1Prod(test/dense/Tensordot/GatherV2_1:output:0%test/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
test/dense/Tensordot/Prod_1
 test/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 test/dense/Tensordot/concat/axisç
test/dense/Tensordot/concatConcatV2"test/dense/Tensordot/free:output:0"test/dense/Tensordot/axes:output:0)test/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
test/dense/Tensordot/concat¸
test/dense/Tensordot/stackPack"test/dense/Tensordot/Prod:output:0$test/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
test/dense/Tensordot/stackÍ
test/dense/Tensordot/transpose	Transposetest/rnn/transpose_1:y:0$test/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
test/dense/Tensordot/transposeË
test/dense/Tensordot/ReshapeReshape"test/dense/Tensordot/transpose:y:0#test/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
test/dense/Tensordot/ReshapeÊ
test/dense/Tensordot/MatMulMatMul%test/dense/Tensordot/Reshape:output:0+test/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
test/dense/Tensordot/MatMul
test/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:S2
test/dense/Tensordot/Const_2
"test/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"test/dense/Tensordot/concat_1/axisô
test/dense/Tensordot/concat_1ConcatV2&test/dense/Tensordot/GatherV2:output:0%test/dense/Tensordot/Const_2:output:0+test/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
test/dense/Tensordot/concat_1Å
test/dense/TensordotReshape%test/dense/Tensordot/MatMul:product:0&test/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
test/dense/Tensordot­
!test/dense/BiasAdd/ReadVariableOpReadVariableOp*test_dense_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype02#
!test/dense/BiasAdd/ReadVariableOp¼
test/dense/BiasAddBiasAddtest/dense/Tensordot:output:0)test/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
test/dense/BiasAddó
IdentityIdentitytest/dense/BiasAdd:output:0"^test/dense/BiasAdd/ReadVariableOp$^test/dense/Tensordot/ReadVariableOp ^test/embedding/embedding_lookup<^test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp;^test/rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp=^test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp>^test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp=^test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp?^test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp^test/rnn/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2F
!test/dense/BiasAdd/ReadVariableOp!test/dense/BiasAdd/ReadVariableOp2J
#test/dense/Tensordot/ReadVariableOp#test/dense/Tensordot/ReadVariableOp2B
test/embedding/embedding_lookuptest/embedding/embedding_lookup2z
;test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp;test/rnn/stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2x
:test/rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:test/rnn/stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2|
<test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp<test/rnn/stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2~
=test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp=test/rnn/stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2|
<test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp<test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2
>test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp>test/rnn/stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2 
test/rnn/whiletest/rnn/while:Y U
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
á

§
G__inference_embedding_layer_call_and_return_conditional_losses_61368027

inputs,
embedding_lookup_61368021:	S
identity¢embedding_lookupf
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Cast
embedding_lookupResourceGatherembedding_lookup_61368021Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*,
_class"
 loc:@embedding_lookup/61368021*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupù
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*,
_class"
 loc:@embedding_lookup/61368021*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identityª
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
§
while_cond_61368825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_placeholder_4
while_placeholder_5
while_less_strided_slice_16
2while_while_cond_61368825___redundant_placeholder06
2while_while_cond_61368825___redundant_placeholder16
2while_while_cond_61368825___redundant_placeholder26
2while_while_cond_61368825___redundant_placeholder36
2while_while_cond_61368825___redundant_placeholder46
2while_while_cond_61368825___redundant_placeholder56
2while_while_cond_61368825___redundant_placeholder6
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesx
v: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: :::::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :	

_output_shapes
:


A__inference_rnn_layer_call_and_return_conditional_losses_61369327

inputsN
:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource:
P
<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource:
J
;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource:	P
<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource:
R
>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource:
L
=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource:	
identity¢2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp¢1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp¢3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp¢4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp¢3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp¢5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1a
zeros_2/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/mul/yr
zeros_2/mulMulstrided_slice:output:0zeros_2/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_2/mulc
zeros_2/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_2/Less/yo
zeros_2/LessLesszeros_2/mul:z:0zeros_2/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_2/Lessg
zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_2/packed/1
zeros_2/packedPackstrided_slice:output:0zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_2/packedc
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Const~
zeros_2Fillzeros_2/packed:output:0zeros_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_2a
zeros_3/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/mul/yr
zeros_3/mulMulstrided_slice:output:0zeros_3/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_3/mulc
zeros_3/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_3/Less/yo
zeros_3/LessLesszeros_3/mul:z:0zeros_3/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_3/Lessg
zeros_3/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
zeros_3/packed/1
zeros_3/packedPackstrided_slice:output:0zeros_3/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_3/packedc
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Const~
zeros_3Fillzeros_3/packed:output:0zeros_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
zeros_3u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¢
+stacked_rnn_cells/lstm_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/ones_like/Shape
+stacked_rnn_cells/lstm_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+stacked_rnn_cells/lstm_cell/ones_like/Constõ
%stacked_rnn_cells/lstm_cell/ones_likeFill4stacked_rnn_cells/lstm_cell/ones_like/Shape:output:04stacked_rnn_cells/lstm_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/ones_like
)stacked_rnn_cells/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2+
)stacked_rnn_cells/lstm_cell/dropout/Constð
'stacked_rnn_cells/lstm_cell/dropout/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:02stacked_rnn_cells/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell/dropout/Mul´
)stacked_rnn_cells/lstm_cell/dropout/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2+
)stacked_rnn_cells/lstm_cell/dropout/Shape¨
@stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform2stacked_rnn_cells/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ý³2B
@stacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform­
2stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>24
2stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y¯
0stacked_rnn_cells/lstm_cell/dropout/GreaterEqualGreaterEqualIstacked_rnn_cells/lstm_cell/dropout/random_uniform/RandomUniform:output:0;stacked_rnn_cells/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0stacked_rnn_cells/lstm_cell/dropout/GreaterEqualÔ
(stacked_rnn_cells/lstm_cell/dropout/CastCast4stacked_rnn_cells/lstm_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stacked_rnn_cells/lstm_cell/dropout/Castë
)stacked_rnn_cells/lstm_cell/dropout/Mul_1Mul+stacked_rnn_cells/lstm_cell/dropout/Mul:z:0,stacked_rnn_cells/lstm_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_1/Constö
)stacked_rnn_cells/lstm_cell/dropout_1/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_1/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_1/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_1/Shape®
Bstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ö2D
Bstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_1/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_1/CastCast6stacked_rnn_cells/lstm_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_1/Castó
+stacked_rnn_cells/lstm_cell/dropout_1/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_1/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_1/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_2/Constö
)stacked_rnn_cells/lstm_cell/dropout_2/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_2/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_2/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_2/Shape®
Bstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2±±2D
Bstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_2/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_2/CastCast6stacked_rnn_cells/lstm_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_2/Castó
+stacked_rnn_cells/lstm_cell/dropout_2/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_2/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_2/Mul_1
+stacked_rnn_cells/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell/dropout_3/Constö
)stacked_rnn_cells/lstm_cell/dropout_3/MulMul.stacked_rnn_cells/lstm_cell/ones_like:output:04stacked_rnn_cells/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell/dropout_3/Mul¸
+stacked_rnn_cells/lstm_cell/dropout_3/ShapeShape.stacked_rnn_cells/lstm_cell/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell/dropout_3/Shape®
Bstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2à2D
Bstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell/dropout_3/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell/dropout_3/CastCast6stacked_rnn_cells/lstm_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell/dropout_3/Castó
+stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Mul-stacked_rnn_cells/lstm_cell/dropout_3/Mul:z:0.stacked_rnn_cells/lstm_cell/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell/dropout_3/Mul_1Å
stacked_rnn_cells/lstm_cell/mulMulstrided_slice_2:output:0-stacked_rnn_cells/lstm_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/mulã
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpReadVariableOp:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOpå
"stacked_rnn_cells/lstm_cell/MatMulMatMul#stacked_rnn_cells/lstm_cell/mul:z:09stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/MatMulé
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOpÖ
$stacked_rnn_cells/lstm_cell/MatMul_1MatMulzeros:output:0;stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell/MatMul_1Ü
stacked_rnn_cells/lstm_cell/addAddV2,stacked_rnn_cells/lstm_cell/MatMul:product:0.stacked_rnn_cells/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
stacked_rnn_cells/lstm_cell/addá
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOpé
#stacked_rnn_cells/lstm_cell/BiasAddBiasAdd#stacked_rnn_cells/lstm_cell/add:z:0:stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/BiasAdd
+stacked_rnn_cells/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+stacked_rnn_cells/lstm_cell/split/split_dim³
!stacked_rnn_cells/lstm_cell/splitSplit4stacked_rnn_cells/lstm_cell/split/split_dim:output:0,stacked_rnn_cells/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2#
!stacked_rnn_cells/lstm_cell/split´
#stacked_rnn_cells/lstm_cell/SigmoidSigmoid*stacked_rnn_cells/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell/Sigmoid¸
%stacked_rnn_cells/lstm_cell/Sigmoid_1Sigmoid*stacked_rnn_cells/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_1½
!stacked_rnn_cells/lstm_cell/mul_1Mul)stacked_rnn_cells/lstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_1«
 stacked_rnn_cells/lstm_cell/TanhTanh*stacked_rnn_cells/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 stacked_rnn_cells/lstm_cell/TanhÏ
!stacked_rnn_cells/lstm_cell/mul_2Mul'stacked_rnn_cells/lstm_cell/Sigmoid:y:0$stacked_rnn_cells/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_2Ð
!stacked_rnn_cells/lstm_cell/add_1AddV2%stacked_rnn_cells/lstm_cell/mul_1:z:0%stacked_rnn_cells/lstm_cell/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/add_1¸
%stacked_rnn_cells/lstm_cell/Sigmoid_2Sigmoid*stacked_rnn_cells/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell/Sigmoid_2ª
"stacked_rnn_cells/lstm_cell/Tanh_1Tanh%stacked_rnn_cells/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell/Tanh_1Ó
!stacked_rnn_cells/lstm_cell/mul_3Mul)stacked_rnn_cells/lstm_cell/Sigmoid_2:y:0&stacked_rnn_cells/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell/mul_3³
-stacked_rnn_cells/lstm_cell_1/ones_like/ShapeShape%stacked_rnn_cells/lstm_cell/mul_3:z:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Shape£
-stacked_rnn_cells/lstm_cell_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-stacked_rnn_cells/lstm_cell_1/ones_like/Constý
'stacked_rnn_cells/lstm_cell_1/ones_likeFill6stacked_rnn_cells/lstm_cell_1/ones_like/Shape:output:06stacked_rnn_cells/lstm_cell_1/ones_like/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/ones_like
+stacked_rnn_cells/lstm_cell_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2-
+stacked_rnn_cells/lstm_cell_1/dropout/Constø
)stacked_rnn_cells/lstm_cell_1/dropout/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:04stacked_rnn_cells/lstm_cell_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stacked_rnn_cells/lstm_cell_1/dropout/Mulº
+stacked_rnn_cells/lstm_cell_1/dropout/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2-
+stacked_rnn_cells/lstm_cell_1/dropout/Shape®
Bstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniformRandomUniform4stacked_rnn_cells/lstm_cell_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2òÜ2D
Bstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform±
4stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>26
4stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y·
2stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualGreaterEqualKstacked_rnn_cells/lstm_cell_1/dropout/random_uniform/RandomUniform:output:0=stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqualÚ
*stacked_rnn_cells/lstm_cell_1/dropout/CastCast6stacked_rnn_cells/lstm_cell_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stacked_rnn_cells/lstm_cell_1/dropout/Castó
+stacked_rnn_cells/lstm_cell_1/dropout/Mul_1Mul-stacked_rnn_cells/lstm_cell_1/dropout/Mul:z:0.stacked_rnn_cells/lstm_cell_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_1/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_1/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_1/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Shape´
Dstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_1/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2§ô2F
Dstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_1/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_1/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_1/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_1/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_1/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_1/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_2/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_2/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_2/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_2/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Shape³
Dstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_2/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2²2F
Dstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_2/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_2/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_2/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_2/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_2/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_2/Mul_1£
-stacked_rnn_cells/lstm_cell_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Constþ
+stacked_rnn_cells/lstm_cell_1/dropout_3/MulMul0stacked_rnn_cells/lstm_cell_1/ones_like:output:06stacked_rnn_cells/lstm_cell_1/dropout_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+stacked_rnn_cells/lstm_cell_1/dropout_3/Mul¾
-stacked_rnn_cells/lstm_cell_1/dropout_3/ShapeShape0stacked_rnn_cells/lstm_cell_1/ones_like:output:0*
T0*
_output_shapes
:2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Shape´
Dstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformRandomUniform6stacked_rnn_cells/lstm_cell_1/dropout_3/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed22F
Dstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniformµ
6stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>28
6stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y¿
4stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualGreaterEqualMstacked_rnn_cells/lstm_cell_1/dropout_3/random_uniform/RandomUniform:output:0?stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqualà
,stacked_rnn_cells/lstm_cell_1/dropout_3/CastCast8stacked_rnn_cells/lstm_cell_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,stacked_rnn_cells/lstm_cell_1/dropout_3/Castû
-stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Mul/stacked_rnn_cells/lstm_cell_1/dropout_3/Mul:z:00stacked_rnn_cells/lstm_cell_1/dropout_3/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-stacked_rnn_cells/lstm_cell_1/dropout_3/Mul_1Ø
!stacked_rnn_cells/lstm_cell_1/mulMul%stacked_rnn_cells/lstm_cell/mul_3:z:0/stacked_rnn_cells/lstm_cell_1/dropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/mulé
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpReadVariableOp<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype025
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOpí
$stacked_rnn_cells/lstm_cell_1/MatMulMatMul%stacked_rnn_cells/lstm_cell_1/mul:z:0;stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/MatMulï
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpReadVariableOp>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype027
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOpÞ
&stacked_rnn_cells/lstm_cell_1/MatMul_1MatMulzeros_2:output:0=stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stacked_rnn_cells/lstm_cell_1/MatMul_1ä
!stacked_rnn_cells/lstm_cell_1/addAddV2.stacked_rnn_cells/lstm_cell_1/MatMul:product:00stacked_rnn_cells/lstm_cell_1/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!stacked_rnn_cells/lstm_cell_1/addç
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpReadVariableOp=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype026
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOpñ
%stacked_rnn_cells/lstm_cell_1/BiasAddBiasAdd%stacked_rnn_cells/lstm_cell_1/add:z:0<stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/BiasAdd 
-stacked_rnn_cells/lstm_cell_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-stacked_rnn_cells/lstm_cell_1/split/split_dim»
#stacked_rnn_cells/lstm_cell_1/splitSplit6stacked_rnn_cells/lstm_cell_1/split/split_dim:output:0.stacked_rnn_cells/lstm_cell_1/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2%
#stacked_rnn_cells/lstm_cell_1/splitº
%stacked_rnn_cells/lstm_cell_1/SigmoidSigmoid,stacked_rnn_cells/lstm_cell_1/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%stacked_rnn_cells/lstm_cell_1/Sigmoid¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_1Ã
#stacked_rnn_cells/lstm_cell_1/mul_1Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_1:y:0zeros_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_1±
"stacked_rnn_cells/lstm_cell_1/TanhTanh,stacked_rnn_cells/lstm_cell_1/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"stacked_rnn_cells/lstm_cell_1/Tanh×
#stacked_rnn_cells/lstm_cell_1/mul_2Mul)stacked_rnn_cells/lstm_cell_1/Sigmoid:y:0&stacked_rnn_cells/lstm_cell_1/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_2Ø
#stacked_rnn_cells/lstm_cell_1/add_1AddV2'stacked_rnn_cells/lstm_cell_1/mul_1:z:0'stacked_rnn_cells/lstm_cell_1/mul_2:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/add_1¾
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2Sigmoid,stacked_rnn_cells/lstm_cell_1/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stacked_rnn_cells/lstm_cell_1/Sigmoid_2°
$stacked_rnn_cells/lstm_cell_1/Tanh_1Tanh'stacked_rnn_cells/lstm_cell_1/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stacked_rnn_cells/lstm_cell_1/Tanh_1Û
#stacked_rnn_cells/lstm_cell_1/mul_3Mul+stacked_rnn_cells/lstm_cell_1/Sigmoid_2:y:0(stacked_rnn_cells/lstm_cell_1/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#stacked_rnn_cells/lstm_cell_1/mul_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterè
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0zeros_2:output:0zeros_3:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0:stacked_rnn_cells_lstm_cell_matmul_readvariableop_resource<stacked_rnn_cells_lstm_cell_matmul_1_readvariableop_resource;stacked_rnn_cells_lstm_cell_biasadd_readvariableop_resource<stacked_rnn_cells_lstm_cell_1_matmul_readvariableop_resource>stacked_rnn_cells_lstm_cell_1_matmul_1_readvariableop_resource=stacked_rnn_cells_lstm_cell_1_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*|
_output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *(
_read_only_resource_inputs


*
bodyR
while_body_61369134*
condR
while_cond_61369133*{
output_shapesj
h: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
transpose_1½
IdentityIdentitytranspose_1:y:03^stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2^stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp4^stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp5^stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4^stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp6^stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : 2h
2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2stacked_rnn_cells/lstm_cell/BiasAdd/ReadVariableOp2f
1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp1stacked_rnn_cells/lstm_cell/MatMul/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp3stacked_rnn_cells/lstm_cell/MatMul_1/ReadVariableOp2l
4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp4stacked_rnn_cells/lstm_cell_1/BiasAdd/ReadVariableOp2j
3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp3stacked_rnn_cells/lstm_cell_1/MatMul/ReadVariableOp2n
5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp5stacked_rnn_cells/lstm_cell_1/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
í
B__inference_test_layer_call_and_return_conditional_losses_61366670

inputs%
embedding_61366373:	S 
rnn_61366620:
 
rnn_61366622:

rnn_61366624:	 
rnn_61366626:
 
rnn_61366628:

rnn_61366630:	!
dense_61366664:	S
dense_61366666:S
identity¢dense/StatefulPartitionedCall¢!embedding/StatefulPartitionedCall¢rnn/StatefulPartitionedCall
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_61366373*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_613663722#
!embedding/StatefulPartitionedCallö
rnn/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0rnn_61366620rnn_61366622rnn_61366624rnn_61366626rnn_61366628rnn_61366630*
Tin
	2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_613666192
rnn/StatefulPartitionedCall¹
dense/StatefulPartitionedCallStatefulPartitionedCall$rnn/StatefulPartitionedCall:output:0dense_61366664dense_61366666*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_613666632
dense/StatefulPartitionedCallé
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^rnn/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
D
input_19
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿF
dense=
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿStensorflow/serving/predict:
¦<
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
X__call__
*Y&call_and_return_all_conditional_losses
Z_default_save_signature"è9
_tf_keras_networkÌ9{"name": "test", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "test", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 83, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "StackedRNNCells", "config": {"name": "stacked_rnn_cells", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}}]}}}, "name": "rnn", "inbound_nodes": [[["embedding", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 83, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["rnn", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "test", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 83, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 1}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "StackedRNNCells", "config": {"name": "stacked_rnn_cells", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 6}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 10}]}, "shared_object_id": 11}}, "name": "rnn", "inbound_nodes": [[["embedding", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 83, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["rnn", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
ï"ì
_tf_keras_input_layerÌ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}



embeddings
trainable_variables
regularization_losses
	variables
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"æ
_tf_keras_layerÌ{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 83, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 1}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}}
Þ
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"µ
_tf_keras_rnn_layer{"name": "rnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "RNN", "config": {"name": "rnn", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "cell": {"class_name": "StackedRNNCells", "config": {"name": "stacked_rnn_cells", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 6}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 10}]}, "shared_object_id": 11}}, "inbound_nodes": [[["embedding", 0, 0, {}]]], "shared_object_id": 12, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 18}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
þ

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 83, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["rnn", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}
 "
trackable_list_wrapper
_

0
1
2
3
4
5
 6
7
8"
trackable_list_wrapper
_

0
1
2
3
4
5
 6
7
8"
trackable_list_wrapper
Ê
!layer_metrics
"non_trainable_variables
#layer_regularization_losses

$layers
regularization_losses
	variables
%metrics
trainable_variables
X__call__
Z_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
aserving_default"
signature_map
':%	S2embedding/embeddings
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
­
trainable_variables
&layer_metrics
'non_trainable_variables

(layers
regularization_losses
	variables
)metrics
*layer_regularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object

	+cells
,trainable_variables
-regularization_losses
.	variables
/	keras_api
b__call__
*c&call_and_return_all_conditional_losses"ê
_tf_keras_layerÐ{"name": "stacked_rnn_cells", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "StackedRNNCells", "config": {"name": "stacked_rnn_cells", "trainable": true, "dtype": "float32", "cells": [{"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 6}, {"class_name": "LSTMCell", "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 10}]}, "shared_object_id": 11}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
¹
0layer_metrics
1non_trainable_variables

2states
3layer_regularization_losses

4layers
regularization_losses
	variables
5metrics
trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:	S2dense/kernel
:S2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
trainable_variables
6layer_metrics
7non_trainable_variables

8layers
regularization_losses
	variables
9metrics
:layer_regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
::8
2&rnn/stacked_rnn_cells/lstm_cell/kernel
D:B
20rnn/stacked_rnn_cells/lstm_cell/recurrent_kernel
3:12$rnn/stacked_rnn_cells/lstm_cell/bias
<::
2(rnn/stacked_rnn_cells/lstm_cell_1/kernel
F:D
22rnn/stacked_rnn_cells/lstm_cell_1/recurrent_kernel
5:32&rnn/stacked_rnn_cells/lstm_cell_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
­
,trainable_variables
=layer_metrics
>non_trainable_variables

?layers
-regularization_losses
.	variables
@metrics
Alayer_regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
	
D
state_size

kernel
recurrent_kernel
bias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
d__call__
*e&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 6}
	
I
state_size

kernel
recurrent_kernel
 bias
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
f__call__
*g&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"name": "lstm_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.3, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 10}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
­
Etrainable_variables
Nlayer_metrics
Onon_trainable_variables

Players
Fregularization_losses
G	variables
Qmetrics
Rlayer_regularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
­
Jtrainable_variables
Slayer_metrics
Tnon_trainable_variables

Ulayers
Kregularization_losses
L	variables
Vmetrics
Wlayer_regularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ê2ç
'__inference_test_layer_call_fn_61366691
'__inference_test_layer_call_fn_61367307
'__inference_test_layer_call_fn_61367330
'__inference_test_layer_call_fn_61367209À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_test_layer_call_and_return_conditional_losses_61367606
B__inference_test_layer_call_and_return_conditional_losses_61368010
B__inference_test_layer_call_and_return_conditional_losses_61367234
B__inference_test_layer_call_and_return_conditional_losses_61367259À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
#__inference__wrapped_model_61365361¿
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª */¢,
*'
input_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ö2Ó
,__inference_embedding_layer_call_fn_61368017¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_embedding_layer_call_and_return_conditional_losses_61368027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
&__inference_rnn_layer_call_fn_61368044
&__inference_rnn_layer_call_fn_61368061
&__inference_rnn_layer_call_fn_61368078
&__inference_rnn_layer_call_fn_61368095æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ø2õ
A__inference_rnn_layer_call_and_return_conditional_losses_61368339
A__inference_rnn_layer_call_and_return_conditional_losses_61368711
A__inference_rnn_layer_call_and_return_conditional_losses_61368955
A__inference_rnn_layer_call_and_return_conditional_losses_61369327æ
Ý²Ù
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_dense_layer_call_fn_61369336¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_layer_call_and_return_conditional_losses_61369366¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
&__inference_signature_wrapper_61367284input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ç2Ä
4__inference_stacked_rnn_cells_layer_call_fn_61369395
4__inference_stacked_rnn_cells_layer_call_fn_61369424Õ
Ì²È
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ý2ú
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61369492
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61369624Õ
Ì²È
FullArgSpec@
args85
jself
jinputs
jstates
j	constants

jtraining
varargs
 
varkwjkwargs
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ä2Á¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ª
#__inference__wrapped_model_61365361	
 9¢6
/¢,
*'
input_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ":ª7
5
dense,)
denseÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS¾
C__inference_dense_layer_call_and_return_conditional_losses_61369366w=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
 
(__inference_dense_layer_call_fn_61369336j=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS½
G__inference_embedding_layer_call_and_return_conditional_losses_61368027r
8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_embedding_layer_call_fn_61368017e
8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÙ
A__inference_rnn_layer_call_and_return_conditional_losses_61368339 T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ù
A__inference_rnn_layer_call_and_return_conditional_losses_61368711 T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
A__inference_rnn_layer_call_and_return_conditional_losses_61368955 M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
A__inference_rnn_layer_call_and_return_conditional_losses_61369327 M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ±
&__inference_rnn_layer_call_fn_61368044 T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ±
&__inference_rnn_layer_call_fn_61368061 T¢Q
J¢G
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
&__inference_rnn_layer_call_fn_61368078 M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
&__inference_rnn_layer_call_fn_61368095 M¢J
C¢@
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
&__inference_signature_wrapper_61367284	
 D¢A
¢ 
:ª7
5
input_1*'
input_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ":ª7
5
dense,)
denseÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61369492Ç è¢ä
Ü¢Ø
!
inputsÿÿÿÿÿÿÿÿÿ
ª¢¦
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ
QN
%"

states/1/0ÿÿÿÿÿÿÿÿÿ
%"

states/1/1ÿÿÿÿÿÿÿÿÿ

 
p 
ª "Ñ¢Í
Å¢Á

0/0ÿÿÿÿÿÿÿÿÿ
¢
KH
"
0/1/0/0ÿÿÿÿÿÿÿÿÿ
"
0/1/0/1ÿÿÿÿÿÿÿÿÿ
KH
"
0/1/1/0ÿÿÿÿÿÿÿÿÿ
"
0/1/1/1ÿÿÿÿÿÿÿÿÿ
 
O__inference_stacked_rnn_cells_layer_call_and_return_conditional_losses_61369624Ç è¢ä
Ü¢Ø
!
inputsÿÿÿÿÿÿÿÿÿ
ª¢¦
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ
QN
%"

states/1/0ÿÿÿÿÿÿÿÿÿ
%"

states/1/1ÿÿÿÿÿÿÿÿÿ

 
p
ª "Ñ¢Í
Å¢Á

0/0ÿÿÿÿÿÿÿÿÿ
¢
KH
"
0/1/0/0ÿÿÿÿÿÿÿÿÿ
"
0/1/0/1ÿÿÿÿÿÿÿÿÿ
KH
"
0/1/1/0ÿÿÿÿÿÿÿÿÿ
"
0/1/1/1ÿÿÿÿÿÿÿÿÿ
 ê
4__inference_stacked_rnn_cells_layer_call_fn_61369395± è¢ä
Ü¢Ø
!
inputsÿÿÿÿÿÿÿÿÿ
ª¢¦
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ
QN
%"

states/1/0ÿÿÿÿÿÿÿÿÿ
%"

states/1/1ÿÿÿÿÿÿÿÿÿ

 
p 
ª "»¢·

0ÿÿÿÿÿÿÿÿÿ
¢
GD
 
1/0/0ÿÿÿÿÿÿÿÿÿ
 
1/0/1ÿÿÿÿÿÿÿÿÿ
GD
 
1/1/0ÿÿÿÿÿÿÿÿÿ
 
1/1/1ÿÿÿÿÿÿÿÿÿê
4__inference_stacked_rnn_cells_layer_call_fn_61369424± è¢ä
Ü¢Ø
!
inputsÿÿÿÿÿÿÿÿÿ
ª¢¦
QN
%"

states/0/0ÿÿÿÿÿÿÿÿÿ
%"

states/0/1ÿÿÿÿÿÿÿÿÿ
QN
%"

states/1/0ÿÿÿÿÿÿÿÿÿ
%"

states/1/1ÿÿÿÿÿÿÿÿÿ

 
p
ª "»¢·

0ÿÿÿÿÿÿÿÿÿ
¢
GD
 
1/0/0ÿÿÿÿÿÿÿÿÿ
 
1/0/1ÿÿÿÿÿÿÿÿÿ
GD
 
1/1/0ÿÿÿÿÿÿÿÿÿ
 
1/1/1ÿÿÿÿÿÿÿÿÿÉ
B__inference_test_layer_call_and_return_conditional_losses_61367234	
 A¢>
7¢4
*'
input_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
 É
B__inference_test_layer_call_and_return_conditional_losses_61367259	
 A¢>
7¢4
*'
input_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
 È
B__inference_test_layer_call_and_return_conditional_losses_61367606	
 @¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
 È
B__inference_test_layer_call_and_return_conditional_losses_61368010	
 @¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
  
'__inference_test_layer_call_fn_61366691u	
 A¢>
7¢4
*'
input_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS 
'__inference_test_layer_call_fn_61367209u	
 A¢>
7¢4
*'
input_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
'__inference_test_layer_call_fn_61367307t	
 @¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
'__inference_test_layer_call_fn_61367330t	
 @¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS