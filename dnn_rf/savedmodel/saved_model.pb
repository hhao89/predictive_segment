*
õÎ
9
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "tag*1.4.02v1.4.0-rc1-11-g130a514ä
F
xPlaceholder*
dtype0*
shape:*
_output_shapes
:
T
w/initial_valueConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
e
w
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 

w/AssignAssignww/initial_value*
validate_shape(*
_class

loc:@w*
use_locking(*
T0*
_output_shapes
: 
L
w/readIdentityw*
_class

loc:@w*
T0*
_output_shapes
: 
W
bias/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
h
bias
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 

bias/AssignAssignbiasbias/initial_value*
validate_shape(*
_class
	loc:@bias*
use_locking(*
T0*
_output_shapes
: 
U
	bias/readIdentitybias*
_class
	loc:@bias*
T0*
_output_shapes
: 
Q
Assign/valueConst*
dtype0*
valueB
 *  @@*
_output_shapes
: 

AssignAssignwAssign/value*
validate_shape(*
_class

loc:@w*
use_locking(*
T0*
_output_shapes
: 
8
MulMulxw/read*
T0*
_output_shapes
:
;
yAddMul	bias/read*
T0*
_output_shapes
:
%
initNoOp	^w/Assign^bias/Assign
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_b6c460bb015742f3a914588f3d682d91/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
h
save/SaveV2/tensor_namesConst*
dtype0*
valueBBbiasBw*
_output_shapes
:
g
save/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B *
_output_shapes
:
~
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbiasw*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*

axis *
T0*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
h
save/RestoreV2/tensor_namesConst*
dtype0*
valueBBbias*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignbiassave/RestoreV2*
validate_shape(*
_class
	loc:@bias*
use_locking(*
T0*
_output_shapes
: 
g
save/RestoreV2_1/tensor_namesConst*
dtype0*
valueBBw*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_1Assignwsave/RestoreV2_1*
validate_shape(*
_class

loc:@w*
use_locking(*
T0*
_output_shapes
: 
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"w
	variablesjh
,
w:0w/Assignw/read:02w/initial_value:0
8
bias:0bias/Assignbias/read:02bias/initial_value:0"
trainable_variablesjh
,
w:0w/Assignw/read:02w/initial_value:0
8
bias:0bias/Assignbias/read:02bias/initial_value:0