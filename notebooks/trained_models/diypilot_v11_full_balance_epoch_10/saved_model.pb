
Њ§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8ла

~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel/m
{
#conv2d/kernel/m/Read/ReadVariableOpReadVariableOpconv2d/kernel/m*&
_output_shapes
:*
dtype0
r
conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias/m
k
!conv2d/bias/m/Read/ReadVariableOpReadVariableOpconv2d/bias/m*
_output_shapes
:*
dtype0

conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_1/kernel/m

%conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/m*&
_output_shapes
: *
dtype0
v
conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/bias/m
o
#conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpconv2d_1/bias/m*
_output_shapes
: *
dtype0

conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_2/kernel/m

%conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
v
conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_2/bias/m
o
#conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpconv2d_2/bias/m*
_output_shapes
:@*
dtype0
y
dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense/kernel/m
r
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_output_shapes
:	@*
dtype0
p
dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense/bias/m
i
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_output_shapes
:@*
dtype0
|
dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_1/kernel/m
u
$dense_1/kernel/m/Read/ReadVariableOpReadVariableOpdense_1/kernel/m*
_output_shapes

:@@*
dtype0
t
dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias/m
m
"dense_1/bias/m/Read/ReadVariableOpReadVariableOpdense_1/bias/m*
_output_shapes
:@*
dtype0
|
dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_2/kernel/m
u
$dense_2/kernel/m/Read/ReadVariableOpReadVariableOpdense_2/kernel/m*
_output_shapes

:@*
dtype0
t
dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias/m
m
"dense_2/bias/m/Read/ReadVariableOpReadVariableOpdense_2/bias/m*
_output_shapes
:*
dtype0

conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel/v
{
#conv2d/kernel/v/Read/ReadVariableOpReadVariableOpconv2d/kernel/v*&
_output_shapes
:*
dtype0
r
conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias/v
k
!conv2d/bias/v/Read/ReadVariableOpReadVariableOpconv2d/bias/v*
_output_shapes
:*
dtype0

conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_1/kernel/v

%conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/v*&
_output_shapes
: *
dtype0
v
conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/bias/v
o
#conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpconv2d_1/bias/v*
_output_shapes
: *
dtype0

conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_2/kernel/v

%conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
v
conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_2/bias/v
o
#conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpconv2d_2/bias/v*
_output_shapes
:@*
dtype0
y
dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense/kernel/v
r
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_output_shapes
:	@*
dtype0
p
dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense/bias/v
i
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_output_shapes
:@*
dtype0
|
dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_1/kernel/v
u
$dense_1/kernel/v/Read/ReadVariableOpReadVariableOpdense_1/kernel/v*
_output_shapes

:@@*
dtype0
t
dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias/v
m
"dense_1/bias/v/Read/ReadVariableOpReadVariableOpdense_1/bias/v*
_output_shapes
:@*
dtype0
|
dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_2/kernel/v
u
$dense_2/kernel/v/Read/ReadVariableOpReadVariableOpdense_2/kernel/v*
_output_shapes

:@*
dtype0
t
dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias/v
m
"dense_2/bias/v/Read/ReadVariableOpReadVariableOpdense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
T
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*еS
valueЫSBШS BСS
ё
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
R
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
R
:regularization_losses
;trainable_variables
<	variables
=	keras_api
R
>regularization_losses
?trainable_variables
@	variables
A	keras_api
R
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
R
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
R
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
h

Tkernel
Ubias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
R
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
R
^regularization_losses
_trainable_variables
`	variables
a	keras_api
h

bkernel
cbias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
А
hiter

ibeta_1

jbeta_2
	kdecay
llearning_rate mЧ!mШ*mЩ+mЪ4mЫ5mЬFmЭGmЮTmЯUmаbmбcmв vг!vд*vе+vж4vз5vиFvйGvкTvлUvмbvнcvо
 
V
 0
!1
*2
+3
44
55
F6
G7
T8
U9
b10
c11
V
 0
!1
*2
+3
44
55
F6
G7
T8
U9
b10
c11
­

mlayers
nmetrics
olayer_metrics
regularization_losses
pnon_trainable_variables
qlayer_regularization_losses
trainable_variables
	variables
 
 
 
 
­

rlayers
smetrics
tnon_trainable_variables
ulayer_metrics
regularization_losses
vlayer_regularization_losses
trainable_variables
	variables
 
 
 
­

wlayers
xmetrics
ynon_trainable_variables
zlayer_metrics
regularization_losses
{layer_regularization_losses
trainable_variables
	variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
Ў

|layers
}metrics
~non_trainable_variables
layer_metrics
"regularization_losses
 layer_regularization_losses
#trainable_variables
$	variables
 
 
 
В
layers
metrics
non_trainable_variables
layer_metrics
&regularization_losses
 layer_regularization_losses
'trainable_variables
(	variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
В
layers
metrics
non_trainable_variables
layer_metrics
,regularization_losses
 layer_regularization_losses
-trainable_variables
.	variables
 
 
 
В
layers
metrics
non_trainable_variables
layer_metrics
0regularization_losses
 layer_regularization_losses
1trainable_variables
2	variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
В
layers
metrics
non_trainable_variables
layer_metrics
6regularization_losses
 layer_regularization_losses
7trainable_variables
8	variables
 
 
 
В
layers
metrics
non_trainable_variables
layer_metrics
:regularization_losses
 layer_regularization_losses
;trainable_variables
<	variables
 
 
 
В
layers
metrics
non_trainable_variables
layer_metrics
>regularization_losses
 layer_regularization_losses
?trainable_variables
@	variables
 
 
 
В
layers
 metrics
Ёnon_trainable_variables
Ђlayer_metrics
Bregularization_losses
 Ѓlayer_regularization_losses
Ctrainable_variables
D	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
В
Єlayers
Ѕmetrics
Іnon_trainable_variables
Їlayer_metrics
Hregularization_losses
 Јlayer_regularization_losses
Itrainable_variables
J	variables
 
 
 
В
Љlayers
Њmetrics
Ћnon_trainable_variables
Ќlayer_metrics
Lregularization_losses
 ­layer_regularization_losses
Mtrainable_variables
N	variables
 
 
 
В
Ўlayers
Џmetrics
Аnon_trainable_variables
Бlayer_metrics
Pregularization_losses
 Вlayer_regularization_losses
Qtrainable_variables
R	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

T0
U1
В
Гlayers
Дmetrics
Еnon_trainable_variables
Жlayer_metrics
Vregularization_losses
 Зlayer_regularization_losses
Wtrainable_variables
X	variables
 
 
 
В
Иlayers
Йmetrics
Кnon_trainable_variables
Лlayer_metrics
Zregularization_losses
 Мlayer_regularization_losses
[trainable_variables
\	variables
 
 
 
В
Нlayers
Оmetrics
Пnon_trainable_variables
Рlayer_metrics
^regularization_losses
 Сlayer_regularization_losses
_trainable_variables
`	variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1

b0
c1
В
Тlayers
Уmetrics
Фnon_trainable_variables
Хlayer_metrics
dregularization_losses
 Цlayer_regularization_losses
etrainable_variables
f	variables
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
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
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEconv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEconv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEdense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEdense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEdense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEdense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lambda_inputPlaceholder*1
_output_shapes
:џџџџџџџџџва*
dtype0*&
shape:џџџџџџџџџва
р
StatefulPartitionedCallStatefulPartitionedCallserving_default_lambda_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_signature_wrapper_1203
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp#conv2d/kernel/m/Read/ReadVariableOp!conv2d/bias/m/Read/ReadVariableOp%conv2d_1/kernel/m/Read/ReadVariableOp#conv2d_1/bias/m/Read/ReadVariableOp%conv2d_2/kernel/m/Read/ReadVariableOp#conv2d_2/bias/m/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp$dense_1/kernel/m/Read/ReadVariableOp"dense_1/bias/m/Read/ReadVariableOp$dense_2/kernel/m/Read/ReadVariableOp"dense_2/bias/m/Read/ReadVariableOp#conv2d/kernel/v/Read/ReadVariableOp!conv2d/bias/v/Read/ReadVariableOp%conv2d_1/kernel/v/Read/ReadVariableOp#conv2d_1/bias/v/Read/ReadVariableOp%conv2d_2/kernel/v/Read/ReadVariableOp#conv2d_2/bias/v/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOp$dense_1/kernel/v/Read/ReadVariableOp"dense_1/bias/v/Read/ReadVariableOp$dense_2/kernel/v/Read/ReadVariableOp"dense_2/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*&
f!R
__inference__traced_save_1743

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasiterbeta_1beta_2decaylearning_rateconv2d/kernel/mconv2d/bias/mconv2d_1/kernel/mconv2d_1/bias/mconv2d_2/kernel/mconv2d_2/bias/mdense/kernel/mdense/bias/mdense_1/kernel/mdense_1/bias/mdense_2/kernel/mdense_2/bias/mconv2d/kernel/vconv2d/bias/vconv2d_1/kernel/vconv2d_1/bias/vconv2d_2/kernel/vconv2d_2/bias/vdense/kernel/vdense/bias/vdense_1/kernel/vdense_1/bias/vdense_2/kernel/vdense_2/bias/v*5
Tin.
,2**
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__traced_restore_1878С	
л
{
&__inference_conv2d_1_layer_call_fn_663

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_6532
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ќ
b
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_601

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Љ
A__inference_dense_1_layer_call_and_return_conditional_losses_1528

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ў
d
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_669

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џ

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_1549

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
з
y
$__inference_conv2d_layer_call_fn_629

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_6192
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
№
y
$__inference_dense_layer_call_fn_1481

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
А>
ї
D__inference_sequential_layer_call_and_return_conditional_losses_1340

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  џB2
lambda/truediv/y
lambda/truedivRealDivinputslambda/truediv/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
lambda/truediva
lambda/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lambda/sub/y

lambda/subSublambda/truediv:z:0lambda/sub/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2

lambda/subИ
max_pooling2d/MaxPoolMaxPoollambda/sub:z:0*1
_output_shapes
:џџџџџџџџџв*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpв
conv2d/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџf*
paddingVALID*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЅ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџf2
conv2d/BiasAdds

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџf2

conv2d/EluФ
max_pooling2d_1/MaxPoolMaxPoolconv2d/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ3E*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpй
conv2d_1/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ! *
paddingVALID*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ! 2
conv2d_1/BiasAddx
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ! 2
conv2d_1/EluЦ
max_pooling2d_2/MaxPoolMaxPoolconv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpй
conv2d_2/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/EluЦ
max_pooling2d_3/MaxPoolMaxPoolconv2d_2/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten/Const
flatten/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Identity 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/BiasAdd~
dropout_1/IdentityIdentitydense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_1/Identityh
elu/EluEludropout_1/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
elu/EluЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulelu/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/BiasAdd
dropout_2/IdentityIdentitydense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_2/Identityl
	elu_1/EluEludropout_2/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
	elu_1/EluЅ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulelu_1/Elu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва:::::::::::::Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
с

"__inference_signature_wrapper_1203
lambda_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityЂStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__wrapped_model_5952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџва
&
_user_specified_namelambda_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
л
{
&__inference_conv2d_2_layer_call_fn_697

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_6872
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Љ

Ї
?__inference_conv2d_layer_call_and_return_conditional_losses_619

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Elu
IdentityIdentityElu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

[
?__inference_lambda_layer_call_and_return_conditional_losses_729

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  џB2
	truediv/yu
truedivRealDivinputstruediv/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2	
truedivS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/yj
subSubtruediv:z:0sub/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
sube
IdentityIdentitysub:z:0*
T0*1
_output_shapes
:џџџџџџџџџва2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџва:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs
І
X
<__inference_elu_layer_call_and_return_conditional_losses_866

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

Ј
@__inference_dense_1_layer_call_and_return_conditional_losses_884

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђ
{
&__inference_dense_1_layer_call_fn_1537

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_8842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ў
d
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_703

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

I
-__inference_max_pooling2d_1_layer_call_fn_641

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6352
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_1498

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ю>

D__inference_sequential_layer_call_and_return_conditional_losses_1137

inputs
conv2d_1097
conv2d_1099
conv2d_1_1103
conv2d_1_1105
conv2d_2_1109
conv2d_2_1111

dense_1117

dense_1119
dense_1_1124
dense_1_1126
dense_2_1131
dense_2_1133
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЖ
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџва* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_7292
lambda/PartitionedCallф
max_pooling2d/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџв* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_6012
max_pooling2d/PartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1097conv2d_1099*
Tin
2*
Tout
2*0
_output_shapes
:џџџџџџџџџf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_6192 
conv2d/StatefulPartitionedCall№
max_pooling2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ3E* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6352!
max_pooling2d_1/PartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_1_1103conv2d_1_1105*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_6532"
 conv2d_1/StatefulPartitionedCallђ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_6692!
max_pooling2d_2/PartitionedCall
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_2_1109conv2d_2_1111*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_6872"
 conv2d_2/StatefulPartitionedCallђ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7032!
max_pooling2d_3/PartitionedCallв
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7672
flatten/PartitionedCallЪ
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_7922
dropout/PartitionedCallљ
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_1117
dense_1119*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8152
dense/StatefulPartitionedCallе
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_8482
dropout_1/PartitionedCallП
elu/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_8662
elu/PartitionedCallџ
dense_1/StatefulPartitionedCallStatefulPartitionedCallelu/PartitionedCall:output:0dense_1_1124dense_1_1126*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_8842!
dense_1/StatefulPartitionedCallз
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_9172
dropout_2/PartitionedCallХ
elu_1/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_elu_1_layer_call_and_return_conditional_losses_9352
elu_1/PartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallelu_1/PartitionedCall:output:0dense_2_1131dense_2_1133*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_9532!
dense_2/StatefulPartitionedCallЧ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


)__inference_sequential_layer_call_fn_1164
lambda_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_11372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџва
&
_user_specified_namelambda_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

A
%__inference_lambda_layer_call_fn_1419

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџва* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_7212
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџва:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs
ќ
_
&__inference_dropout_layer_call_fn_1457

inputs
identityЂStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_7872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
]
A__inference_flatten_layer_call_and_return_conditional_losses_1430

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

I
-__inference_max_pooling2d_2_layer_call_fn_675

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_6692
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

[
?__inference_lambda_layer_call_and_return_conditional_losses_721

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  џB2
	truediv/yu
truedivRealDivinputstruediv/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2	
truedivS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/yj
subSubtruediv:z:0sub/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
sube
IdentityIdentitysub:z:0*
T0*1
_output_shapes
:џџџџџџџџџва2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџва:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs
ў

a
B__inference_dropout_2_layer_call_and_return_conditional_losses_912

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
K
л
__inference__wrapped_model_595
lambda_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity
sequential/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  џB2
sequential/lambda/truediv/yБ
sequential/lambda/truedivRealDivlambda_input$sequential/lambda/truediv/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
sequential/lambda/truedivw
sequential/lambda/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sequential/lambda/sub/yВ
sequential/lambda/subSubsequential/lambda/truediv:z:0 sequential/lambda/sub/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
sequential/lambda/subй
 sequential/max_pooling2d/MaxPoolMaxPoolsequential/lambda/sub:z:0*1
_output_shapes
:џџџџџџџџџв*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolЫ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpў
sequential/conv2d/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџf*
paddingVALID*
strides
2
sequential/conv2d/Conv2DТ
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpб
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџf2
sequential/conv2d/BiasAdd
sequential/conv2d/EluElu"sequential/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџf2
sequential/conv2d/Eluх
"sequential/max_pooling2d_1/MaxPoolMaxPool#sequential/conv2d/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ3E*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPoolб
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp
sequential/conv2d_1/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ! *
paddingVALID*
strides
2
sequential/conv2d_1/Conv2DШ
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpи
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ! 2
sequential/conv2d_1/BiasAdd
sequential/conv2d_1/EluElu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ! 2
sequential/conv2d_1/Eluч
"sequential/max_pooling2d_2/MaxPoolMaxPool%sequential/conv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPoolб
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_2/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
sequential/conv2d_2/Conv2DШ
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOpи
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential/conv2d_2/BiasAdd
sequential/conv2d_2/EluElu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
sequential/conv2d_2/Eluч
"sequential/max_pooling2d_3/MaxPoolMaxPool%sequential/conv2d_2/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_3/MaxPool
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
sequential/flatten/ConstЦ
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_3/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential/flatten/Reshape
sequential/dropout/IdentityIdentity#sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential/dropout/IdentityС
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&sequential/dense/MatMul/ReadVariableOpФ
sequential/dense/MatMulMatMul$sequential/dropout/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dense/MatMulП
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpХ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dense/BiasAdd
sequential/dropout_1/IdentityIdentity!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dropout_1/Identity
sequential/elu/EluElu&sequential/dropout_1/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/elu/EluЦ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpЦ
sequential/dense_1/MatMulMatMul sequential/elu/Elu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dense_1/MatMulХ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЭ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dense_1/BiasAddЁ
sequential/dropout_2/IdentityIdentity#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/dropout_2/Identity
sequential/elu_1/EluElu&sequential/dropout_2/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
sequential/elu_1/EluЦ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpШ
sequential/dense_2/MatMulMatMul"sequential/elu_1/Elu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_2/MatMulХ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpЭ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential/dense_2/BiasAddw
IdentityIdentity#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва:::::::::::::_ [
1
_output_shapes
:џџџџџџџџџва
&
_user_specified_namelambda_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ќ

)__inference_sequential_layer_call_fn_1398

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_11372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ќ
a
(__inference_dropout_1_layer_call_fn_1503

inputs
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_8432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

Ј
@__inference_dense_2_layer_call_and_return_conditional_losses_953

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
а>

D__inference_sequential_layer_call_and_return_conditional_losses_1015
lambda_input

conv2d_975

conv2d_977
conv2d_1_981
conv2d_1_983
conv2d_2_987
conv2d_2_989
	dense_995
	dense_997
dense_1_1002
dense_1_1004
dense_2_1009
dense_2_1011
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallМ
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџва* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_7292
lambda/PartitionedCallф
max_pooling2d/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџв* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_6012
max_pooling2d/PartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0
conv2d_975
conv2d_977*
Tin
2*
Tout
2*0
_output_shapes
:џџџџџџџџџf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_6192 
conv2d/StatefulPartitionedCall№
max_pooling2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ3E* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6352!
max_pooling2d_1/PartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_1_981conv2d_1_983*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_6532"
 conv2d_1/StatefulPartitionedCallђ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_6692!
max_pooling2d_2/PartitionedCall
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_2_987conv2d_2_989*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_6872"
 conv2d_2/StatefulPartitionedCallђ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7032!
max_pooling2d_3/PartitionedCallв
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7672
flatten/PartitionedCallЪ
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_7922
dropout/PartitionedCallї
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0	dense_995	dense_997*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8152
dense/StatefulPartitionedCallе
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_8482
dropout_1/PartitionedCallП
elu/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_8662
elu/PartitionedCallџ
dense_1/StatefulPartitionedCallStatefulPartitionedCallelu/PartitionedCall:output:0dense_1_1002dense_1_1004*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_8842!
dense_1/StatefulPartitionedCallз
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_9172
dropout_2/PartitionedCallХ
elu_1/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_elu_1_layer_call_and_return_conditional_losses_9352
elu_1/PartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallelu_1/PartitionedCall:output:0dense_2_1009dense_2_1011*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_9532!
dense_2/StatefulPartitionedCallЧ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџва
&
_user_specified_namelambda_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ш
_
A__inference_dropout_layer_call_and_return_conditional_losses_1452

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

_
@__inference_dropout_layer_call_and_return_conditional_losses_787

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў
d
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_635

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
C
ю
D__inference_sequential_layer_call_and_return_conditional_losses_1063

inputs
conv2d_1023
conv2d_1025
conv2d_1_1029
conv2d_1_1031
conv2d_2_1035
conv2d_2_1037

dense_1043

dense_1045
dense_1_1050
dense_1_1052
dense_2_1057
dense_2_1059
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЖ
lambda/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџва* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_7212
lambda/PartitionedCallф
max_pooling2d/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџв* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_6012
max_pooling2d/PartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1023conv2d_1025*
Tin
2*
Tout
2*0
_output_shapes
:џџџџџџџџџf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_6192 
conv2d/StatefulPartitionedCall№
max_pooling2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ3E* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6352!
max_pooling2d_1/PartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_1_1029conv2d_1_1031*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_6532"
 conv2d_1/StatefulPartitionedCallђ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_6692!
max_pooling2d_2/PartitionedCall
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_2_1035conv2d_2_1037*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_6872"
 conv2d_2/StatefulPartitionedCallђ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7032!
max_pooling2d_3/PartitionedCallв
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7672
flatten/PartitionedCallт
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_7872!
dropout/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dense_1043
dense_1045*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8152
dense/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_8432#
!dropout_1/StatefulPartitionedCallЧ
elu/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_8662
elu/PartitionedCallџ
dense_1/StatefulPartitionedCallStatefulPartitionedCallelu/PartitionedCall:output:0dense_1_1050dense_1_1052*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_8842!
dense_1/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_9122#
!dropout_2/StatefulPartitionedCallЭ
elu_1/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_elu_1_layer_call_and_return_conditional_losses_9352
elu_1/PartitionedCall
dense_2/StatefulPartitionedCallStatefulPartitionedCallelu_1/PartitionedCall:output:0dense_2_1057dense_2_1059*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_9532!
dense_2/StatefulPartitionedCallБ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
C
ч
C__inference_sequential_layer_call_and_return_conditional_losses_970
lambda_input

conv2d_743

conv2d_745
conv2d_1_749
conv2d_1_751
conv2d_2_755
conv2d_2_757
	dense_826
	dense_828
dense_1_895
dense_1_897
dense_2_964
dense_2_966
identityЂconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallМ
lambda/PartitionedCallPartitionedCalllambda_input*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџва* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_7212
lambda/PartitionedCallф
max_pooling2d/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџв* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_6012
max_pooling2d/PartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0
conv2d_743
conv2d_745*
Tin
2*
Tout
2*0
_output_shapes
:џџџџџџџџџf*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_conv2d_layer_call_and_return_conditional_losses_6192 
conv2d/StatefulPartitionedCall№
max_pooling2d_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ3E* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_6352!
max_pooling2d_1/PartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_1_749conv2d_1_751*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ! *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_1_layer_call_and_return_conditional_losses_6532"
 conv2d_1/StatefulPartitionedCallђ
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_6692!
max_pooling2d_2/PartitionedCall
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_2_755conv2d_2_757*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_conv2d_2_layer_call_and_return_conditional_losses_6872"
 conv2d_2/StatefulPartitionedCallђ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7032!
max_pooling2d_3/PartitionedCallв
flatten/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7672
flatten/PartitionedCallт
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_7872!
dropout/StatefulPartitionedCallџ
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0	dense_826	dense_828*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_8152
dense/StatefulPartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_8432#
!dropout_1/StatefulPartitionedCallЧ
elu/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_8662
elu/PartitionedCall§
dense_1/StatefulPartitionedCallStatefulPartitionedCallelu/PartitionedCall:output:0dense_1_895dense_1_897*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_8842!
dense_1/StatefulPartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_9122#
!dropout_2/StatefulPartitionedCallЭ
elu_1/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_elu_1_layer_call_and_return_conditional_losses_9352
elu_1/PartitionedCallџ
dense_2/StatefulPartitionedCallStatefulPartitionedCallelu_1/PartitionedCall:output:0dense_2_964dense_2_966*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_9532!
dense_2/StatefulPartitionedCallБ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџва
&
_user_specified_namelambda_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
>
"__inference_elu_layer_call_fn_1518

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*E
f@R>
<__inference_elu_layer_call_and_return_conditional_losses_8662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Х
`
B__inference_dropout_1_layer_call_and_return_conditional_losses_848

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

І
>__inference_dense_layer_call_and_return_conditional_losses_815

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
№
B
&__inference_dropout_layer_call_fn_1462

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_7922
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
[
?__inference_elu_1_layer_call_and_return_conditional_losses_1569

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


)__inference_sequential_layer_call_fn_1090
lambda_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_10632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџва
&
_user_specified_namelambda_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ќ
a
(__inference_dropout_2_layer_call_fn_1559

inputs
identityЂStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_9122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

`
A__inference_dropout_layer_call_and_return_conditional_losses_1447

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
юЏ

 __inference__traced_restore_1878
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias%
!assignvariableop_8_dense_1_kernel#
assignvariableop_9_dense_1_bias&
"assignvariableop_10_dense_2_kernel$
 assignvariableop_11_dense_2_bias
assignvariableop_12_iter
assignvariableop_13_beta_1
assignvariableop_14_beta_2
assignvariableop_15_decay%
!assignvariableop_16_learning_rate'
#assignvariableop_17_conv2d_kernel_m%
!assignvariableop_18_conv2d_bias_m)
%assignvariableop_19_conv2d_1_kernel_m'
#assignvariableop_20_conv2d_1_bias_m)
%assignvariableop_21_conv2d_2_kernel_m'
#assignvariableop_22_conv2d_2_bias_m&
"assignvariableop_23_dense_kernel_m$
 assignvariableop_24_dense_bias_m(
$assignvariableop_25_dense_1_kernel_m&
"assignvariableop_26_dense_1_bias_m(
$assignvariableop_27_dense_2_kernel_m&
"assignvariableop_28_dense_2_bias_m'
#assignvariableop_29_conv2d_kernel_v%
!assignvariableop_30_conv2d_bias_v)
%assignvariableop_31_conv2d_1_kernel_v'
#assignvariableop_32_conv2d_1_bias_v)
%assignvariableop_33_conv2d_2_kernel_v'
#assignvariableop_34_conv2d_2_bias_v&
"assignvariableop_35_dense_kernel_v$
 assignvariableop_36_dense_bias_v(
$assignvariableop_37_dense_1_kernel_v&
"assignvariableop_38_dense_1_bias_v(
$assignvariableop_39_dense_2_kernel_v&
"assignvariableop_40_dense_2_bias_v
identity_42ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1Ъ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*ж
valueЬBЩ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesр
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesћ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*К
_output_shapesЇ
Є:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0	*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterIdentity_12:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_beta_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOpassignvariableop_14_beta_2Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOpassignvariableop_15_decayIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOp!assignvariableop_16_learning_rateIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv2d_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv2d_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19
AssignVariableOp_19AssignVariableOp%assignvariableop_19_conv2d_1_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_1_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21
AssignVariableOp_21AssignVariableOp%assignvariableop_21_conv2d_2_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22
AssignVariableOp_22AssignVariableOp#assignvariableop_22_conv2d_2_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23
AssignVariableOp_23AssignVariableOp"assignvariableop_23_dense_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25
AssignVariableOp_25AssignVariableOp$assignvariableop_25_dense_1_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_1_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_2_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_2_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29
AssignVariableOp_29AssignVariableOp#assignvariableop_29_conv2d_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30
AssignVariableOp_30AssignVariableOp!assignvariableop_30_conv2d_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31
AssignVariableOp_31AssignVariableOp%assignvariableop_31_conv2d_1_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv2d_1_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33
AssignVariableOp_33AssignVariableOp%assignvariableop_33_conv2d_2_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34
AssignVariableOp_34AssignVariableOp#assignvariableop_34_conv2d_2_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35
AssignVariableOp_35AssignVariableOp"assignvariableop_35_dense_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36
AssignVariableOp_36AssignVariableOp assignvariableop_36_dense_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37
AssignVariableOp_37AssignVariableOp$assignvariableop_37_dense_1_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38
AssignVariableOp_38AssignVariableOp"assignvariableop_38_dense_1_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39
AssignVariableOp_39AssignVariableOp$assignvariableop_39_dense_2_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40
AssignVariableOp_40AssignVariableOp"assignvariableop_40_dense_2_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpф
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41ё
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*Л
_input_shapesЉ
І: :::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: 
ш
@
$__inference_elu_1_layer_call_fn_1574

inputs
identity
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_elu_1_layer_call_and_return_conditional_losses_9352
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

Ї
?__inference_dense_layer_call_and_return_conditional_losses_1472

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
И
\
@__inference_flatten_layer_call_and_return_conditional_losses_767

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

\
@__inference_lambda_layer_call_and_return_conditional_losses_1414

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  џB2
	truediv/yu
truedivRealDivinputstruediv/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2	
truedivS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/yj
subSubtruediv:z:0sub/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
sube
IdentityIdentitysub:z:0*
T0*1
_output_shapes
:џџџџџџџџџва2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџва:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs
Х
`
B__inference_dropout_2_layer_call_and_return_conditional_losses_917

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ч
^
@__inference_dropout_layer_call_and_return_conditional_losses_792

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ

Љ
A__inference_conv2d_1_layer_call_and_return_conditional_losses_653

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
Elu
IdentityIdentityElu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
џY
ї
D__inference_sequential_layer_call_and_return_conditional_losses_1282

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  џB2
lambda/truediv/y
lambda/truedivRealDivinputslambda/truediv/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
lambda/truediva
lambda/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
lambda/sub/y

lambda/subSublambda/truediv:z:0lambda/sub/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2

lambda/subИ
max_pooling2d/MaxPoolMaxPoollambda/sub:z:0*1
_output_shapes
:џџџџџџџџџв*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpв
conv2d/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџf*
paddingVALID*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЅ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџf2
conv2d/BiasAdds

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџf2

conv2d/EluФ
max_pooling2d_1/MaxPoolMaxPoolconv2d/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ3E*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOpй
conv2d_1/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ! *
paddingVALID*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ! 2
conv2d_1/BiasAddx
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ! 2
conv2d_1/EluЦ
max_pooling2d_2/MaxPoolMaxPoolconv2d_1/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpй
conv2d_2/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpЌ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv2d_2/EluЦ
max_pooling2d_3/MaxPoolMaxPoolconv2d_2/Elu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten/Const
flatten/ReshapeReshape max_pooling2d_3/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeЭ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>2 
dropout/dropout/GreaterEqual/yп
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/Mul_1 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstЁ
dropout_1/dropout/MulMuldense/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_1/dropout/Mulx
dropout_1/dropout/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeв
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yц
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout_1/dropout/CastЂ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_1/dropout/Mul_1h
elu/EluEludropout_1/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
elu/EluЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulelu/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_1/BiasAddw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/ConstЃ
dropout_2/dropout/MulMuldense_1/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_2/dropout/Mulz
dropout_2/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeв
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yц
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout_2/dropout/CastЂ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout_2/dropout/Mul_1l
	elu_1/EluEludropout_2/dropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
	elu_1/EluЅ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulelu_1/Elu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddl
IdentityIdentitydense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва:::::::::::::Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

G
+__inference_max_pooling2d_layer_call_fn_607

inputs
identityШ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_6012
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џ

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_1493

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ќ

)__inference_sequential_layer_call_fn_1369

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_10632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:џџџџџџџџџва::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

Љ
A__inference_dense_2_layer_call_and_return_conditional_losses_1584

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
с\
щ
__inference__traced_save_1743
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop.
*savev2_conv2d_kernel_m_read_readvariableop,
(savev2_conv2d_bias_m_read_readvariableop0
,savev2_conv2d_1_kernel_m_read_readvariableop.
*savev2_conv2d_1_bias_m_read_readvariableop0
,savev2_conv2d_2_kernel_m_read_readvariableop.
*savev2_conv2d_2_bias_m_read_readvariableop-
)savev2_dense_kernel_m_read_readvariableop+
'savev2_dense_bias_m_read_readvariableop/
+savev2_dense_1_kernel_m_read_readvariableop-
)savev2_dense_1_bias_m_read_readvariableop/
+savev2_dense_2_kernel_m_read_readvariableop-
)savev2_dense_2_bias_m_read_readvariableop.
*savev2_conv2d_kernel_v_read_readvariableop,
(savev2_conv2d_bias_v_read_readvariableop0
,savev2_conv2d_1_kernel_v_read_readvariableop.
*savev2_conv2d_1_bias_v_read_readvariableop0
,savev2_conv2d_2_kernel_v_read_readvariableop.
*savev2_conv2d_2_bias_v_read_readvariableop-
)savev2_dense_kernel_v_read_readvariableop+
'savev2_dense_bias_v_read_readvariableop/
+savev2_dense_1_kernel_v_read_readvariableop-
)savev2_dense_1_bias_v_read_readvariableop/
+savev2_dense_2_kernel_v_read_readvariableop-
)savev2_dense_2_bias_v_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c22bbe4754a7478594c3a6b02acf236b/part2	
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameФ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*ж
valueЬBЩ)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesк
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop*savev2_conv2d_kernel_m_read_readvariableop(savev2_conv2d_bias_m_read_readvariableop,savev2_conv2d_1_kernel_m_read_readvariableop*savev2_conv2d_1_bias_m_read_readvariableop,savev2_conv2d_2_kernel_m_read_readvariableop*savev2_conv2d_2_bias_m_read_readvariableop)savev2_dense_kernel_m_read_readvariableop'savev2_dense_bias_m_read_readvariableop+savev2_dense_1_kernel_m_read_readvariableop)savev2_dense_1_bias_m_read_readvariableop+savev2_dense_2_kernel_m_read_readvariableop)savev2_dense_2_bias_m_read_readvariableop*savev2_conv2d_kernel_v_read_readvariableop(savev2_conv2d_bias_v_read_readvariableop,savev2_conv2d_1_kernel_v_read_readvariableop*savev2_conv2d_1_bias_v_read_readvariableop,savev2_conv2d_2_kernel_v_read_readvariableop*savev2_conv2d_2_bias_v_read_readvariableop)savev2_dense_kernel_v_read_readvariableop'savev2_dense_bias_v_read_readvariableop+savev2_dense_1_kernel_v_read_readvariableop)savev2_dense_1_bias_v_read_readvariableop+savev2_dense_2_kernel_v_read_readvariableop)savev2_dense_2_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesќ
љ: ::: : : @:@:	@:@:@@:@:@:: : : : : ::: : : @:@:	@:@:@@:@:@:::: : : @:@:	@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:%$!

_output_shapes
:	@: %

_output_shapes
:@:$& 

_output_shapes

:@@: '

_output_shapes
:@:$( 

_output_shapes

:@: )

_output_shapes
::*

_output_shapes
: 
ў

a
B__inference_dropout_1_layer_call_and_return_conditional_losses_843

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeД
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ћ

Љ
A__inference_conv2d_2_layer_call_and_return_conditional_losses_687

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddo
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
Elu
IdentityIdentityElu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ :::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
№
D
(__inference_dropout_1_layer_call_fn_1508

inputs
identityЁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_1_layer_call_and_return_conditional_losses_8482
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

I
-__inference_max_pooling2d_3_layer_call_fn_709

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7032
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№
D
(__inference_dropout_2_layer_call_fn_1564

inputs
identityЁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dropout_2_layer_call_and_return_conditional_losses_9172
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ї
Y
=__inference_elu_layer_call_and_return_conditional_losses_1513

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ц
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_1554

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

A
%__inference_lambda_layer_call_fn_1424

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*1
_output_shapes
:џџџџџџџџџва* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*H
fCRA
?__inference_lambda_layer_call_and_return_conditional_losses_7292
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџва:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs
Ј
Z
>__inference_elu_1_layer_call_and_return_conditional_losses_935

inputs
identityK
EluEluinputs*
T0*'
_output_shapes
:џџџџџџџџџ@2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ђ
{
&__inference_dense_2_layer_call_fn_1593

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_9532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ў
B
&__inference_flatten_layer_call_fn_1435

inputs
identity 
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_7672
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

\
@__inference_lambda_layer_call_and_return_conditional_losses_1406

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  џB2
	truediv/yu
truedivRealDivinputstruediv/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2	
truedivS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/yj
subSubtruediv:z:0sub/y:output:0*
T0*1
_output_shapes
:џџџџџџџџџва2
sube
IdentityIdentitysub:z:0*
T0*1
_output_shapes
:џџџџџџџџџва2

Identity"
identityIdentity:output:0*0
_input_shapes
:џџџџџџџџџва:Y U
1
_output_shapes
:џџџџџџџџџва
 
_user_specified_nameinputs"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
O
lambda_input?
serving_default_lambda_input:0џџџџџџџџџва;
dense_20
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Х
f
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
п_default_save_signature
+р&call_and_return_all_conditional_losses
с__call__"Мa
_tf_keras_sequentiala{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 210, 848, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwwAAAB8AGQBGwBkAhgAUwApA05nAAAAAADgX0BnAAAAAAAA\n8D+pACkB2gF4cgEAAAByAQAAAPofPGlweXRob24taW5wdXQtMTQtYjk5M2IzOGFlZmVkPtoIPGxh\nbWJkYT4HAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": [210, 848, 3], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 3]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "ELU", "config": {"name": "elu", "trainable": true, "dtype": "float32", "alpha": 1.0}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "ELU", "config": {"name": "elu_1", "trainable": true, "dtype": "float32", "alpha": 1.0}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 210, 848, 3]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 210, 848, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 210, 848, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwwAAAB8AGQBGwBkAhgAUwApA05nAAAAAADgX0BnAAAAAAAA\n8D+pACkB2gF4cgEAAAByAQAAAPofPGlweXRob24taW5wdXQtMTQtYjk5M2IzOGFlZmVkPtoIPGxh\nbWJkYT4HAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": [210, 848, 3], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 3]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "ELU", "config": {"name": "elu", "trainable": true, "dtype": "float32", "alpha": 1.0}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "ELU", "config": {"name": "elu_1", "trainable": true, "dtype": "float32", "alpha": 1.0}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 210, 848, 3]}}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error", "mean_absolute_error"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ђ
regularization_losses
trainable_variables
	variables
	keras_api
+т&call_and_return_all_conditional_losses
у__call__"
_tf_keras_layerї{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 210, 848, 3]}, "stateful": false, "config": {"name": "lambda", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 210, 848, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABDAAAAcwwAAAB8AGQBGwBkAhgAUwApA05nAAAAAADgX0BnAAAAAAAA\n8D+pACkB2gF4cgEAAAByAQAAAPofPGlweXRob24taW5wdXQtMTQtYjk5M2IzOGFlZmVkPtoIPGxh\nbWJkYT4HAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": [210, 848, 3], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
к
regularization_losses
trainable_variables
	variables
	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"Щ
_tf_keras_layerЏ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Р	

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"
_tf_keras_layerџ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [7, 7]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 210, 282, 3]}}
о
&regularization_losses
'trainable_variables
(	variables
)	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"Э
_tf_keras_layerГ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ф	

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
+ъ&call_and_return_all_conditional_losses
ы__call__"
_tf_keras_layer{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 69, 16]}}
о
0regularization_losses
1trainable_variables
2	variables
3	keras_api
+ь&call_and_return_all_conditional_losses
э__call__"Э
_tf_keras_layerГ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ф	

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
+ю&call_and_return_all_conditional_losses
я__call__"
_tf_keras_layer{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 16, 32]}}
о
:regularization_losses
;trainable_variables
<	variables
=	keras_api
+№&call_and_return_all_conditional_losses
ё__call__"Э
_tf_keras_layerГ{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
С
>regularization_losses
?trainable_variables
@	variables
A	keras_api
+ђ&call_and_return_all_conditional_losses
ѓ__call__"А
_tf_keras_layer{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Р
Bregularization_losses
Ctrainable_variables
D	variables
E	keras_api
+є&call_and_return_all_conditional_losses
ѕ__call__"Џ
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Я

Fkernel
Gbias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+і&call_and_return_all_conditional_losses
ї__call__"Ј
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 384]}}
Ф
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+ј&call_and_return_all_conditional_losses
љ__call__"Г
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}

Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
+њ&call_and_return_all_conditional_losses
ћ__call__"
_tf_keras_layerш{"class_name": "ELU", "name": "elu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "elu", "trainable": true, "dtype": "float32", "alpha": 1.0}}
б

Tkernel
Ubias
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
+ќ&call_and_return_all_conditional_losses
§__call__"Њ
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
Ф
Zregularization_losses
[trainable_variables
\	variables
]	keras_api
+ў&call_and_return_all_conditional_losses
џ__call__"Г
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}

^regularization_losses
_trainable_variables
`	variables
a	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerь{"class_name": "ELU", "name": "elu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "elu_1", "trainable": true, "dtype": "float32", "alpha": 1.0}}
а

bkernel
cbias
dregularization_losses
etrainable_variables
f	variables
g	keras_api
+&call_and_return_all_conditional_losses
__call__"Љ
_tf_keras_layer{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
У
hiter

ibeta_1

jbeta_2
	kdecay
llearning_rate mЧ!mШ*mЩ+mЪ4mЫ5mЬFmЭGmЮTmЯUmаbmбcmв vг!vд*vе+vж4vз5vиFvйGvкTvлUvмbvнcvо"
	optimizer
 "
trackable_list_wrapper
v
 0
!1
*2
+3
44
55
F6
G7
T8
U9
b10
c11"
trackable_list_wrapper
v
 0
!1
*2
+3
44
55
F6
G7
T8
U9
b10
c11"
trackable_list_wrapper
Ю

mlayers
nmetrics
olayer_metrics
regularization_losses
pnon_trainable_variables
qlayer_regularization_losses
trainable_variables
	variables
с__call__
п_default_save_signature
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

rlayers
smetrics
tnon_trainable_variables
ulayer_metrics
regularization_losses
vlayer_regularization_losses
trainable_variables
	variables
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

wlayers
xmetrics
ynon_trainable_variables
zlayer_metrics
regularization_losses
{layer_regularization_losses
trainable_variables
	variables
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
':%2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
Б

|layers
}metrics
~non_trainable_variables
layer_metrics
"regularization_losses
 layer_regularization_losses
#trainable_variables
$	variables
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
metrics
non_trainable_variables
layer_metrics
&regularization_losses
 layer_regularization_losses
'trainable_variables
(	variables
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
Е
layers
metrics
non_trainable_variables
layer_metrics
,regularization_losses
 layer_regularization_losses
-trainable_variables
.	variables
ы__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
metrics
non_trainable_variables
layer_metrics
0regularization_losses
 layer_regularization_losses
1trainable_variables
2	variables
э__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
:@2conv2d_2/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
Е
layers
metrics
non_trainable_variables
layer_metrics
6regularization_losses
 layer_regularization_losses
7trainable_variables
8	variables
я__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
metrics
non_trainable_variables
layer_metrics
:regularization_losses
 layer_regularization_losses
;trainable_variables
<	variables
ё__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
metrics
non_trainable_variables
layer_metrics
>regularization_losses
 layer_regularization_losses
?trainable_variables
@	variables
ѓ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
 metrics
Ёnon_trainable_variables
Ђlayer_metrics
Bregularization_losses
 Ѓlayer_regularization_losses
Ctrainable_variables
D	variables
ѕ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
:	@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
Е
Єlayers
Ѕmetrics
Іnon_trainable_variables
Їlayer_metrics
Hregularization_losses
 Јlayer_regularization_losses
Itrainable_variables
J	variables
ї__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Љlayers
Њmetrics
Ћnon_trainable_variables
Ќlayer_metrics
Lregularization_losses
 ­layer_regularization_losses
Mtrainable_variables
N	variables
љ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Ўlayers
Џmetrics
Аnon_trainable_variables
Бlayer_metrics
Pregularization_losses
 Вlayer_regularization_losses
Qtrainable_variables
R	variables
ћ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 :@@2dense_1/kernel
:@2dense_1/bias
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
Е
Гlayers
Дmetrics
Еnon_trainable_variables
Жlayer_metrics
Vregularization_losses
 Зlayer_regularization_losses
Wtrainable_variables
X	variables
§__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Иlayers
Йmetrics
Кnon_trainable_variables
Лlayer_metrics
Zregularization_losses
 Мlayer_regularization_losses
[trainable_variables
\	variables
џ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Нlayers
Оmetrics
Пnon_trainable_variables
Рlayer_metrics
^regularization_losses
 Сlayer_regularization_losses
_trainable_variables
`	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
Е
Тlayers
Уmetrics
Фnon_trainable_variables
Хlayer_metrics
dregularization_losses
 Цlayer_regularization_losses
etrainable_variables
f	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
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
':%2conv2d/kernel/m
:2conv2d/bias/m
):' 2conv2d_1/kernel/m
: 2conv2d_1/bias/m
):' @2conv2d_2/kernel/m
:@2conv2d_2/bias/m
:	@2dense/kernel/m
:@2dense/bias/m
 :@@2dense_1/kernel/m
:@2dense_1/bias/m
 :@2dense_2/kernel/m
:2dense_2/bias/m
':%2conv2d/kernel/v
:2conv2d/bias/v
):' 2conv2d_1/kernel/v
: 2conv2d_1/bias/v
):' @2conv2d_2/kernel/v
:@2conv2d_2/bias/v
:	@2dense/kernel/v
:@2dense/bias/v
 :@@2dense_1/kernel/v
:@2dense_1/bias/v
 :@2dense_2/kernel/v
:2dense_2/bias/v
ы2ш
__inference__wrapped_model_595Х
В
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
annotationsЊ *5Ђ2
0-
lambda_inputџџџџџџџџџва
н2к
D__inference_sequential_layer_call_and_return_conditional_losses_1015
D__inference_sequential_layer_call_and_return_conditional_losses_1282
D__inference_sequential_layer_call_and_return_conditional_losses_1340
C__inference_sequential_layer_call_and_return_conditional_losses_970Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
)__inference_sequential_layer_call_fn_1398
)__inference_sequential_layer_call_fn_1369
)__inference_sequential_layer_call_fn_1164
)__inference_sequential_layer_call_fn_1090Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
Ъ2Ч
@__inference_lambda_layer_call_and_return_conditional_losses_1414
@__inference_lambda_layer_call_and_return_conditional_losses_1406Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
%__inference_lambda_layer_call_fn_1424
%__inference_lambda_layer_call_fn_1419Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ў2Ћ
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_601р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
+__inference_max_pooling2d_layer_call_fn_607р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
?__inference_conv2d_layer_call_and_return_conditional_losses_619з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
$__inference_conv2d_layer_call_fn_629з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
А2­
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_635р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
-__inference_max_pooling2d_1_layer_call_fn_641р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 2
A__inference_conv2d_1_layer_call_and_return_conditional_losses_653з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
&__inference_conv2d_1_layer_call_fn_663з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
А2­
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_669р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
-__inference_max_pooling2d_2_layer_call_fn_675р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 2
A__inference_conv2d_2_layer_call_and_return_conditional_losses_687з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
2
&__inference_conv2d_2_layer_call_fn_697з
В
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
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
А2­
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_703р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
-__inference_max_pooling2d_3_layer_call_fn_709р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
ы2ш
A__inference_flatten_layer_call_and_return_conditional_losses_1430Ђ
В
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
annotationsЊ *
 
а2Э
&__inference_flatten_layer_call_fn_1435Ђ
В
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
annotationsЊ *
 
Р2Н
A__inference_dropout_layer_call_and_return_conditional_losses_1452
A__inference_dropout_layer_call_and_return_conditional_losses_1447Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
&__inference_dropout_layer_call_fn_1462
&__inference_dropout_layer_call_fn_1457Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_1472Ђ
В
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
annotationsЊ *
 
Ю2Ы
$__inference_dense_layer_call_fn_1481Ђ
В
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
annotationsЊ *
 
Ф2С
C__inference_dropout_1_layer_call_and_return_conditional_losses_1493
C__inference_dropout_1_layer_call_and_return_conditional_losses_1498Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
(__inference_dropout_1_layer_call_fn_1508
(__inference_dropout_1_layer_call_fn_1503Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ч2ф
=__inference_elu_layer_call_and_return_conditional_losses_1513Ђ
В
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
annotationsЊ *
 
Ь2Щ
"__inference_elu_layer_call_fn_1518Ђ
В
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
annotationsЊ *
 
ы2ш
A__inference_dense_1_layer_call_and_return_conditional_losses_1528Ђ
В
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
annotationsЊ *
 
а2Э
&__inference_dense_1_layer_call_fn_1537Ђ
В
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
annotationsЊ *
 
Ф2С
C__inference_dropout_2_layer_call_and_return_conditional_losses_1549
C__inference_dropout_2_layer_call_and_return_conditional_losses_1554Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
(__inference_dropout_2_layer_call_fn_1564
(__inference_dropout_2_layer_call_fn_1559Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
щ2ц
?__inference_elu_1_layer_call_and_return_conditional_losses_1569Ђ
В
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
annotationsЊ *
 
Ю2Ы
$__inference_elu_1_layer_call_fn_1574Ђ
В
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
annotationsЊ *
 
ы2ш
A__inference_dense_2_layer_call_and_return_conditional_losses_1584Ђ
В
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
annotationsЊ *
 
а2Э
&__inference_dense_2_layer_call_fn_1593Ђ
В
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
annotationsЊ *
 
6B4
"__inference_signature_wrapper_1203lambda_inputЅ
__inference__wrapped_model_595 !*+45FGTUbc?Ђ<
5Ђ2
0-
lambda_inputџџџџџџџџџва
Њ "1Њ.
,
dense_2!
dense_2џџџџџџџџџж
A__inference_conv2d_1_layer_call_and_return_conditional_losses_653*+IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Ў
&__inference_conv2d_1_layer_call_fn_663*+IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ж
A__inference_conv2d_2_layer_call_and_return_conditional_losses_68745IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Ў
&__inference_conv2d_2_layer_call_fn_69745IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
?__inference_conv2d_layer_call_and_return_conditional_losses_619 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ќ
$__inference_conv2d_layer_call_fn_629 !IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЁ
A__inference_dense_1_layer_call_and_return_conditional_losses_1528\TU/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 y
&__inference_dense_1_layer_call_fn_1537OTU/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@Ё
A__inference_dense_2_layer_call_and_return_conditional_losses_1584\bc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 y
&__inference_dense_2_layer_call_fn_1593Obc/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ 
?__inference_dense_layer_call_and_return_conditional_losses_1472]FG0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 x
$__inference_dense_layer_call_fn_1481PFG0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ѓ
C__inference_dropout_1_layer_call_and_return_conditional_losses_1493\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "%Ђ"

0џџџџџџџџџ@
 Ѓ
C__inference_dropout_1_layer_call_and_return_conditional_losses_1498\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "%Ђ"

0џџџџџџџџџ@
 {
(__inference_dropout_1_layer_call_fn_1503O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@{
(__inference_dropout_1_layer_call_fn_1508O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@Ѓ
C__inference_dropout_2_layer_call_and_return_conditional_losses_1549\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "%Ђ"

0џџџџџџџџџ@
 Ѓ
C__inference_dropout_2_layer_call_and_return_conditional_losses_1554\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "%Ђ"

0џџџџџџџџџ@
 {
(__inference_dropout_2_layer_call_fn_1559O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p
Њ "џџџџџџџџџ@{
(__inference_dropout_2_layer_call_fn_1564O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ@
p 
Њ "џџџџџџџџџ@Ѓ
A__inference_dropout_layer_call_and_return_conditional_losses_1447^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 Ѓ
A__inference_dropout_layer_call_and_return_conditional_losses_1452^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 {
&__inference_dropout_layer_call_fn_1457Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ{
&__inference_dropout_layer_call_fn_1462Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
?__inference_elu_1_layer_call_and_return_conditional_losses_1569X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 s
$__inference_elu_1_layer_call_fn_1574K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@
=__inference_elu_layer_call_and_return_conditional_losses_1513X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ@
 q
"__inference_elu_layer_call_fn_1518K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ@І
A__inference_flatten_layer_call_and_return_conditional_losses_1430a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "&Ђ#

0џџџџџџџџџ
 ~
&__inference_flatten_layer_call_fn_1435T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "џџџџџџџџџИ
@__inference_lambda_layer_call_and_return_conditional_losses_1406tAЂ>
7Ђ4
*'
inputsџџџџџџџџџва

 
p
Њ "/Ђ,
%"
0џџџџџџџџџва
 И
@__inference_lambda_layer_call_and_return_conditional_losses_1414tAЂ>
7Ђ4
*'
inputsџџџџџџџџџва

 
p 
Њ "/Ђ,
%"
0џџџџџџџџџва
 
%__inference_lambda_layer_call_fn_1419gAЂ>
7Ђ4
*'
inputsџџџџџџџџџва

 
p
Њ ""џџџџџџџџџва
%__inference_lambda_layer_call_fn_1424gAЂ>
7Ђ4
*'
inputsџџџџџџџџџва

 
p 
Њ ""џџџџџџџџџваы
H__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_635RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 У
-__inference_max_pooling2d_1_layer_call_fn_641RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџы
H__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_669RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 У
-__inference_max_pooling2d_2_layer_call_fn_675RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџы
H__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_703RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 У
-__inference_max_pooling2d_3_layer_call_fn_709RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџщ
F__inference_max_pooling2d_layer_call_and_return_conditional_losses_601RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 С
+__inference_max_pooling2d_layer_call_fn_607RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЦ
D__inference_sequential_layer_call_and_return_conditional_losses_1015~ !*+45FGTUbcGЂD
=Ђ:
0-
lambda_inputџџџџџџџџџва
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Р
D__inference_sequential_layer_call_and_return_conditional_losses_1282x !*+45FGTUbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџва
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Р
D__inference_sequential_layer_call_and_return_conditional_losses_1340x !*+45FGTUbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџва
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Х
C__inference_sequential_layer_call_and_return_conditional_losses_970~ !*+45FGTUbcGЂD
=Ђ:
0-
lambda_inputџџџџџџџџџва
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
)__inference_sequential_layer_call_fn_1090q !*+45FGTUbcGЂD
=Ђ:
0-
lambda_inputџџџџџџџџџва
p

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_1164q !*+45FGTUbcGЂD
=Ђ:
0-
lambda_inputџџџџџџџџџва
p 

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_1369k !*+45FGTUbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџва
p

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_1398k !*+45FGTUbcAЂ>
7Ђ4
*'
inputsџџџџџџџџџва
p 

 
Њ "џџџџџџџџџЙ
"__inference_signature_wrapper_1203 !*+45FGTUbcOЂL
Ђ 
EЊB
@
lambda_input0-
lambda_inputџџџџџџџџџва"1Њ.
,
dense_2!
dense_2џџџџџџџџџ