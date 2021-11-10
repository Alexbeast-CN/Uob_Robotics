% This is the model of the 5-DOF robot Lynxmotion

deg = pi/180;

% Main parameters
d1 = 300;
l1 = 400;
l2 = 500;
l3 = 200;

% Link([theta, d, a alpha])
L(1) = Revolute('d',d1,'a',0,'alpha',-pi/2,'qlim', [-180 180]*deg);

L(2) = Revolute('d',0,'a',l1,'alpha',0,'qlim', [0 180]*deg);

L(3) = Revolute('d',0,'a',l2,'alpha',0,'qlim', [0 180]*deg);

L(4) = Revolute('d',0,'a',0,'alpha',-pi/2,'qlim', [0 180]*deg);

L(5) = Revolute('d',0,'a',l3,'alpha',0,'qlim', [-180 180]*deg);

%
% some useful poses
%
qz = [0 0 0 0 0 0]; % zero angles, L shaped pose
qr = [0 pi/2 -pi/2 0 0 0]; % ready pose, arm up
qs = [0 0 -pi/2 0 0 0];
qn=[0 pi/4 pi 0 pi/4  0];

Lynxmotion = SerialLink(L, 'name', 'Lynxmotion', ...
    'configs', {'qz', qz, 'qr', qr, 'qs', qs, 'qn', qn});

