% This is the inverse kinematics for the 5-DOF Lynxmotion.
% It's usinddg Pieper's solution.
function Thetas = pieper_solution(T)

    mdl_Lynxmotion
    n = Lynxmotion.n;

    Thetas = zeros(1,n);

    % Main parameters of
    d1 = 300;
    l1 = 400;
    l2 = 500;
    l3 = 200;

    syms theta1 theta2 theta3 theta4 theta5 

    T01 = FKMatrix(0,0,d1,theta1);
    T12 = FKMatrix(-pi/2,0,0,theta2);
    T23 = FKMatrix(0,l1,0,theta3);
    T34 = FKMatrix(0,l2,0,theta4);
    T45 = FKMatrix(-pi/2,0,0,theta5);

    P04 = T01*T12*T23*T34(:,4);

    eqn1 = P04(1,1) == 0;
    eqn2 = P04(2,1) == 0;
    eqn3 = P04(3,1) == 0;
    s = solve([eqn1,eqn2,eqn3],[theta1,theta2,theta3]);
    theta1 = (s.theta1(2,1))
    theta2 = (s.theta2(2,1))
    theta3 = (s.theta3(2,1))

end

function T = FK4IK(alpha, a, d, theta)
    T = [cosd(theta), -sind(theta), 0, a;
         sind(theta)*cosd(sym(alpha)), cosd(theta)*cosd(sym(alpha)), -sind(sym(alpha)), -sind(sym(alpha))*d;
         sind(theta)*sind(sym(alpha)), cosd(theta)*sind(sym(alpha)), cosd(sym(alpha)), cosd(sym(alpha))*d;
         0, 0, 0, 1];
end