% Reference centrelines for the SOFA scene, one CSV per load case.
%   matlab -batch "run('~/sofa/thesis-scenes/matlab/run_cases.m')"
% Writes ../matlab_refs/<case>.csv (soft-part centreline, metres, world/mesh frame).
here = fileparts(mfilename('fullpath'));
refs = fullfile(here, '..', 'matlab_refs');
pose = jsondecode(fileread(fullfile(here, '..', 'poses', 'kocher_patient3.json')));
tip = [0.005 0 -0.010];

% name, gravity [m/s^2], tip force [N], use Kocher pose
cases = {
    'matlab_default',     [0 0 -9.81], tip,     false
    'actuation_only',     [0 0 0],     [0 0 0], false
    'actuation_tipforce', [0 0 0],     tip,     false
    'hanging',            [0 0 9.81],  tip,     false
    'kocher_head0_air',   pose.gravity.supine_head0_air.',  [0 0 0], true
    'kocher_head0_csf',   pose.gravity.supine_head0_csf.',  [0 0 0], true
    'kocher_head30_csf',  pose.gravity.supine_head30_csf.', [0 0 0], true
};
only = {};                                  % e.g. {'kocher_head0_csf'} to run a subset
for c = 1:size(cases, 1)
    if ~isempty(only) && ~any(strcmp(only, cases{c,1})), continue, end
    clear CASE_P CASE_Q CASE_L CASE_BEND CASE_SEC
    CASE_G = cases{c,2}; CASE_F = cases{c,3};
    if cases{c,4}
        CASE_P = pose.base_position_m; CASE_Q = pose.base_quat_wxyz;
    end
    CASE_OUT = fullfile(refs, [cases{c,1} '.csv']);
    run(fullfile(here, 'cosserat_case.m'));
end

% Resized-robot cases: poses/design_*.json (tools/make_design_cases.py)
designs = dir(fullfile(here, '..', 'poses', 'design_*.json'));
for c = 1:numel(designs)
    name = erase(designs(c).name, '.json');
    if ~isempty(only) && ~any(strcmp(only, name)), continue, end
    dz = jsondecode(fileread(fullfile(designs(c).folder, designs(c).name)));
    clear CASE_P CASE_Q CASE_L CASE_BEND CASE_SEC
    CASE_G = dz.gravity(:).'; CASE_F = [0 0 0];
    CASE_P = dz.base_position_m; CASE_Q = dz.base_quat_wxyz;
    CASE_L = [dz.sections.L_m];
    CASE_BEND = [[dz.sections.bend_rad].', [dz.sections.plane_rad].'];
    if isfield(dz.sections, 'radius_m')
        CASE_SEC = [[dz.sections.radius_m].', [dz.sections.chamberOffset_m].', [dz.sections.chamberRadius_m].'];
    end
    CASE_OUT = fullfile(refs, [name '.csv']);
    run(fullfile(here, 'cosserat_case.m'));
end
