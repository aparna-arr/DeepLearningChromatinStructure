#!/usr/bin/env perl
use warnings;
use strict;

## A quick perl script to scrape AUC results out of
## the somewhat messy log files
## And generate the LaTeX formatted table of 
## Hyperparameter search results
## I'm more comfortable with Perl for parsing so that's
## the only reason why it's not in Python!

my ($loglist) = @ARGV;
die "usage: $0 <file with list of .log files>\n" unless @ARGV == 1;

# get log filenames from the loglist
my @logfiles;

open(LOGLIST, "<", $loglist) or die "could not open $loglist:$!\n";

while(<LOGLIST>) {
	my $line = $_;
	chomp $line;

	push(@logfiles,$line);
}

close LOGLIST;

my @results;

# iterate over each log file, extracting information
foreach my $file (@logfiles) {
	my $devROC = 0;
	my $trainROC = 0;

	# get the parameter values
	my ($model, $learn, $wdecay, $minibatch, $epochs) = $file =~ /model(.+)-0_learn_(.+)_weight_decay_(.+)_minibatch_(\d+)_epochs_(\d+)_hyperparam/;

	# treat conv and FC model outputs differently
	# due to a change in their logfiles
	if ($file =~ /conv/) { 
		open(LOG, "<", $file) or die "could not open $file: $!\n";
	
		while(<LOG>) {
			my $line = $_;
			chomp $line;
			if ($line =~ /ROC/) {
				if ($line =~ /Dev/) {
					($devROC) = $line =~ /Dev.+: (\S+)$/;
				} else {
					($trainROC) = $line =~ /Train.+: (\S+)$/;
				}
			}
		}

		close LOG;

	} else {
		open(LOG, "<", $file) or die "could not open $file: $!\n";
	
		while(<LOG>) {
			my $line = $_;
			chomp $line;
			if ($line =~ /ROC/) {
				if ($line =~ /Test/) {		
					($devROC) = $line =~ /Test.*: (\S+)$/;
				} else {
                        		($trainROC) = $line =~ /Train.*: (\S+)$/;
				}
			}
		}

		close LOG;
	}

	# organize the results
	push(@results, {
		model => $model,
		learn => $learn,
		wdecay => $wdecay,
		minibatch => $minibatch,
		epochs => $epochs,
		trainROC => $trainROC,
		devROC => $devROC
	});	
}

# sort in order of descending AUC
#
my @sortDevROC = sort{ $b->{devROC} <=> $a->{devROC} } @results;

# quickly print the best value
warn "best dev ROC is $sortDevROC[0]{devROC} from model $sortDevROC[0]{model} with learning rate $sortDevROC[0]{learn} wdecay $sortDevROC[0]{wdecay} minibatch $sortDevROC[0]{minibatch}";

# generate latex table
my $header = "\\begin{tabular}{||c|c|c|c|c|c|c||}\n\\hline\n";

$header .= "Model & Learning Rate & Weight Decay & Minibatch Size & Epochs & Train Score & Dev Score \\\\\n\\hline\\hline\n";
foreach my $hashref (@sortDevROC){
	my $model = $hashref->{model};
	my $learn = $hashref->{learn};
	my $wdecay = $hashref->{wdecay};
	my $minibatch = $hashref->{minibatch};
	my $epochs = $hashref->{epochs};
	my $trainROC = $hashref->{trainROC};
	my $devROC = $hashref->{devROC};

	$learn = sprintf("%.10f", $learn);
	$wdecay = sprintf("%.10f", $wdecay);

	$learn =~ s/\.?0*$//;
	$wdecay =~ s/\.?0*$//;

	if ($model =~ /conv/) {
		$model =~ s/conv/Conv /;
		$model = "Model " . $model;
	}
	else {
		$model = "Model FC " . $model;
	}

	$header .= "$model & $learn & $wdecay & $minibatch & $epochs & $trainROC & $devROC \\\\\n\\hline\n";
}

open(OUT, ">", "latex.txt") or die "could not open outfile!:$!\n";
print OUT $header;
close OUT;
