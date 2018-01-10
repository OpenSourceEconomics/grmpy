#!/usr/bin/env python
# encoding: utf-8
# Carlos Rafael Giani, 2006
# Thomas Nagy, 2010-2016 (ita)

"""
Unit testing system for C/C++/D providing test execution:

* in parallel, by using ``waf -j``
* partial (only the tests that have changed) or full (by using ``waf --alltests``)

The tests are declared by adding the **test** feature to programs::

	def options(opt):
		opt.load('compiler_cxx waf_unit_test')
	def configure(conf):
		conf.load('compiler_cxx waf_unit_test')
	def build(bld):
		bld(features='cxx cxxprogram test', source='main.cpp', target='app')
		# or
		bld.program(features='test', source='main2.cpp', target='app2')

When the build is executed, the program 'test' will be built and executed without arguments.
The success/failure is detected by looking at the return code. The status and the standard output/error
are stored on the build context.

The results can be displayed by registering a callback function. Here is how to call
the predefined callback::

	def build(bld):
		bld(features='cxx cxxprogram test', source='main.c', target='app')
		from waflib.Tools import waf_unit_test
		bld.add_post_fun(waf_unit_test.summary)
"""

import os
from waflib.TaskGen import feature, after_method, taskgen_method
from waflib import Utils, Task, Logs, Options
from waflib.Tools import ccroot
testlock = Utils.threading.Lock()

@feature('test')
@after_method('apply_link', 'process_use')
def make_test(self):
	"""Create the unit test task. There can be only one unit test task by task generator."""
	if not getattr(self, 'link_task', None):
		return

	tsk = self.create_task('utest', self.link_task.outputs)
	if getattr(self, 'ut_str', None):
		self.ut_run, lst = Task.compile_fun(self.ut_str, shell=getattr(self, 'ut_shell', False))
		tsk.vars = lst + tsk.vars

	if getattr(self, 'ut_cwd', None):
		if isinstance(self.ut_cwd, str):
			# we want a Node instance
			if os.path.isabs(self.ut_cwd):
				self.ut_cwd = self.bld.root.make_node(self.ut_cwd)
			else:
				self.ut_cwd = self.path.make_node(self.ut_cwd)
	else:
		self.ut_cwd = tsk.inputs[0].parent

	if not hasattr(self, 'ut_paths'):
		paths = []
		for x in self.tmp_use_sorted:
			try:
				y = self.bld.get_tgen_by_name(x).link_task
			except AttributeError:
				pass
			else:
				if not isinstance(y, ccroot.stlink_task):
					paths.append(y.outputs[0].parent.abspath())
		self.ut_paths = os.pathsep.join(paths) + os.pathsep

	if not hasattr(self, 'ut_env'):
		self.ut_env = dct = dict(os.environ)
		def add_path(var):
			dct[var] = self.ut_paths + dct.get(var,'')
		if Utils.is_win32:
			add_path('PATH')
		elif Utils.unversioned_sys_platform() == 'darwin':
			add_path('DYLD_LIBRARY_PATH')
			add_path('LD_LIBRARY_PATH')
		else:
			add_path('LD_LIBRARY_PATH')

@taskgen_method
def add_test_results(self, tup):
	"""Override and return tup[1] to interrupt the build immediately if a test does not run"""
	Logs.debug("ut: %r", tup)
	self.utest_result = tup
	try:
		self.bld.utest_results.append(tup)
	except AttributeError:
		self.bld.utest_results = [tup]

class utest(Task.Task):
	"""
	Execute a unit test
	"""
	color = 'PINK'
	after = ['vnum', 'inst']
	vars = []

	def runnable_status(self):
		"""
		Always execute the task if `waf --alltests` was used or no
		tests if ``waf --notests`` was used
		"""
		if getattr(Options.options, 'no_tests', False):
			return Task.SKIP_ME

		ret = super(utest, self).runnable_status()
		if ret == Task.SKIP_ME:
			if getattr(Options.options, 'all_tests', False):
				return Task.RUN_ME
		return ret

	def get_test_env(self):
		"""
		In general, tests may require any library built anywhere in the project.
		Override this method if fewer paths are needed
		"""
		return self.generator.ut_env

	def post_run(self):
		super(utest, self).post_run()
		if getattr(Options.options, 'clear_failed_tests', False) and self.waf_unit_test_results[1]:
			self.generator.bld.task_sigs[self.uid()] = None

	def run(self):
		"""
		Execute the test. The execution is always successful, and the results
		are stored on ``self.generator.bld.utest_results`` for postprocessing.

		Override ``add_test_results`` to interrupt the build
		"""
		if hasattr(self.generator, 'ut_run'):
			return self.generator.ut_run(self)

		# TODO ut_exec, ut_fun, ut_cmd should be considered obsolete
		self.ut_exec = getattr(self.generator, 'ut_exec', [self.inputs[0].abspath()])
		if getattr(self.generator, 'ut_fun', None):
			self.generator.ut_fun(self)
		testcmd = getattr(self.generator, 'ut_cmd', False) or getattr(Options.options, 'testcmd', False)
		if testcmd:
			self.ut_exec = (testcmd % ' '.join(self.ut_exec)).split(' ')

		return self.exec_command(self.ut_exec)

	def exec_command(self, cmd, **kw):
		Logs.debug('runner: %r', cmd)
		proc = Utils.subprocess.Popen(cmd, cwd=self.get_cwd().abspath(), env=self.get_test_env(),
			stderr=Utils.subprocess.PIPE, stdout=Utils.subprocess.PIPE)
		(stdout, stderr) = proc.communicate()
		self.waf_unit_test_results = tup = (self.inputs[0].abspath(), proc.returncode, stdout, stderr)
		testlock.acquire()
		try:
			return self.generator.add_test_results(tup)
		finally:
			testlock.release()

	def get_cwd(self):
		return self.generator.ut_cwd

def summary(bld):
	"""
	Display an execution summary::

		def build(bld):
			bld(features='cxx cxxprogram test', source='main.c', target='app')
			from waflib.Tools import waf_unit_test
			bld.add_post_fun(waf_unit_test.summary)
	"""
	lst = getattr(bld, 'utest_results', [])
	if lst:
		Logs.pprint('CYAN', 'execution summary')

		total = len(lst)
		tfail = len([x for x in lst if x[1]])

		Logs.pprint('CYAN', '  tests that pass %d/%d' % (total-tfail, total))
		for (f, code, out, err) in lst:
			if not code:
				Logs.pprint('CYAN', '    %s' % f)

		Logs.pprint('CYAN', '  tests that fail %d/%d' % (tfail, total))
		for (f, code, out, err) in lst:
			if code:
				Logs.pprint('CYAN', '    %s' % f)

def set_exit_code(bld):
	"""
	If any of the tests fail waf will exit with that exit code.
	This is useful if you have an automated build system which need
	to report on errors from the tests.
	You may use it like this:

		def build(bld):
			bld(features='cxx cxxprogram test', source='main.c', target='app')
			from waflib.Tools import waf_unit_test
			bld.add_post_fun(waf_unit_test.set_exit_code)
	"""
	lst = getattr(bld, 'utest_results', [])
	for (f, code, out, err) in lst:
		if code:
			msg = []
			if out:
				msg.append('stdout:%s%s' % (os.linesep, out.decode('utf-8')))
			if err:
				msg.append('stderr:%s%s' % (os.linesep, err.decode('utf-8')))
			bld.fatal(os.linesep.join(msg))


def options(opt):
	"""
	Provide the ``--alltests``, ``--notests`` and ``--testcmd`` command-line options.
	"""
	opt.add_option('--notests', action='store_true', default=False, help='Exec no unit tests', dest='no_tests')
	opt.add_option('--alltests', action='store_true', default=False, help='Exec all unit tests', dest='all_tests')
	opt.add_option('--clear-failed', action='store_true', default=False, help='Force failed unit tests to run again next time', dest='clear_failed_tests')
	opt.add_option('--testcmd', action='store', default=False,
	 help = 'Run the unit tests using the test-cmd string'
	 ' example "--test-cmd="valgrind --error-exitcode=1'
	 ' %s" to run under valgrind', dest='testcmd')

