import { Link } from '@tanstack/react-router'
import { ShieldCheck, Sparkles, TrendingUp, Users } from 'lucide-react'
import { UserAuthForm } from './components/user-auth-form'

export function SignIn2() {
  return (
    <div className='relative min-h-screen w-full lg:grid lg:grid-cols-2'>
      {/* Left Column: Sign-in Form */}
      <div className='flex min-h-screen flex-col justify-between p-6 sm:p-12 lg:p-16'>
        {/* Top Branding */}
        <div className='flex items-center gap-2.5'>
          <div className='flex size-9 items-center justify-center rounded-lg bg-indigo-600 text-lg font-bold text-white shadow-md shadow-indigo-600/30'>
            🏢
          </div>
          <div>
            <div className='font-bold text-foreground text-base tracking-tight leading-none'>Vantara Retail</div>
            <div className='text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mt-0.5'>
              Enterprise CRM Suite
            </div>
          </div>
        </div>

        {/* Center Form Area */}
        <div className='mx-auto my-auto w-full max-w-sm space-y-6'>
          <div className='space-y-1.5'>
            <h1 className='text-2xl font-bold tracking-tight text-foreground'>Sign In to CRM</h1>
            <p className='text-xs text-muted-foreground'>
              Enter your corporate credentials to access customer behavior & retention intelligence.
            </p>
          </div>

          <UserAuthForm />

          <div className='text-center text-xs text-muted-foreground'>
            Don&apos;t have an account?{' '}
            <Link to='/sign-up-2' className='font-semibold text-indigo-600 hover:text-indigo-500 underline underline-offset-4 dark:text-indigo-400'>
              Sign Up (2 Col)
            </Link>
            {' '}or{' '}
            <Link to='/sign-up' className='font-semibold text-indigo-600 hover:text-indigo-500 underline underline-offset-4 dark:text-indigo-400'>
              Simple Sign Up
            </Link>
          </div>
        </div>

        {/* Footer Disclaimer */}
        <div className='text-center text-[11px] text-muted-foreground'>
          Protected by Vantara Enterprise IAM • SOC2 Type II Certified
        </div>
      </div>

      {/* Right Column: High-Craft Enterprise Showcase */}
      <div className='hidden lg:flex flex-col justify-between bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 p-12 text-white border-l border-border/40 relative overflow-hidden'>
        {/* Ambient background glow */}
        <div className='absolute -top-24 -right-24 size-96 rounded-full bg-indigo-600/20 blur-3xl pointer-events-none' />
        <div className='absolute -bottom-24 -left-24 size-96 rounded-full bg-emerald-600/15 blur-3xl pointer-events-none' />

        {/* Top Status */}
        <div className='relative z-10 flex items-center justify-between'>
          <div className='flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium backdrop-blur-md'>
            <span className='size-2 rounded-full bg-emerald-400 animate-pulse' />
            Production Engine v2.4 Active
          </div>
          <div className='text-xs font-mono text-slate-400'>
            5,225 Accounts Synchronized
          </div>
        </div>

        {/* Middle Value Props */}
        <div className='relative z-10 space-y-6 my-auto max-w-lg'>
          <div>
            <div className='inline-flex items-center gap-1.5 rounded-md bg-indigo-500/20 px-2.5 py-1 text-xs font-semibold text-indigo-300 border border-indigo-500/30 mb-3'>
              <Sparkles className='size-3.5' />
              Predictive Customer Intelligence
            </div>
            <h2 className='text-3xl font-extrabold tracking-tight leading-tight text-white'>
              Stop Customer Churn Before It Happens.
            </h2>
            <p className='mt-2 text-sm text-slate-300 leading-relaxed'>
              Vantara Enterprise CRM pairs calibrated machine learning with explainable AI to protect revenue and forecast customer lifetime value with pinpoint accuracy.
            </p>
          </div>

          {/* Feature Highlights Grid */}
          <div className='space-y-3'>
            <div className='flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3.5 backdrop-blur-md'>
              <div className='flex size-8 shrink-0 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400'>
                <ShieldCheck className='size-4' />
              </div>
              <div className='text-xs'>
                <div className='font-semibold text-white'>Calibrated 90-Day Churn Prediction</div>
                <div className='text-slate-300 mt-0.5'>XGBoost classification core evaluated with 5-Fold Stratified CV (80.8% ROC-AUC).</div>
              </div>
            </div>

            <div className='flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3.5 backdrop-blur-md'>
              <div className='flex size-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-400'>
                <TrendingUp className='size-4' />
              </div>
              <div className='text-xs'>
                <div className='font-semibold text-white'>Forward 90-Day CLV Forecasting</div>
                <div className='text-slate-300 mt-0.5'>Regularized Ridge regression predicting future net monetary spend (R² 0.845).</div>
              </div>
            </div>

            <div className='flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3.5 backdrop-blur-md'>
              <div className='flex size-8 shrink-0 items-center justify-center rounded-lg bg-amber-500/20 text-amber-400'>
                <Users className='size-4' />
              </div>
              <div className='text-xs'>
                <div className='font-semibold text-white'>Behavioral RFM Personas & XAI</div>
                <div className='text-slate-300 mt-0.5'>Automated clustering and local SHAP feature drivers explaining root-cause risk.</div>
              </div>
            </div>
          </div>

          {/* Executive Testimonial */}
          <div className='rounded-xl border border-white/10 bg-gradient-to-r from-white/10 to-white/5 p-4 backdrop-blur-md'>
            <p className='text-xs italic text-slate-200 leading-relaxed'>
              &ldquo;Vantara CRM empowered our Customer Success and Retention teams to intervene on £3.3M in high-value at-risk accounts, automating playbooks with immediate revenue impact.&rdquo;
            </p>
            <div className='mt-2.5 flex items-center gap-2 text-xs'>
              <div className='size-6 rounded-full bg-indigo-500 flex items-center justify-center font-bold text-[10px] text-white'>
                VR
              </div>
              <div>
                <span className='font-semibold text-white'>Retention Operations Lead</span>
                <span className='text-slate-400'> • Vantara Retail UK</span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Metrics Bar */}
        <div className='relative z-10 grid grid-cols-3 gap-4 border-t border-white/10 pt-4 text-center'>
          <div>
            <div className='text-lg font-bold text-white font-mono'>80.8%</div>
            <div className='text-[10px] text-slate-400 uppercase tracking-wider'>Test ROC-AUC</div>
          </div>
          <div>
            <div className='text-lg font-bold text-emerald-400 font-mono'>0.845</div>
            <div className='text-[10px] text-slate-400 uppercase tracking-wider'>CLV R² Score</div>
          </div>
          <div>
            <div className='text-lg font-bold text-indigo-400 font-mono'>&lt;15ms</div>
            <div className='text-[10px] text-slate-400 uppercase tracking-wider'>Inference Latency</div>
          </div>
        </div>
      </div>
    </div>
  )
}
