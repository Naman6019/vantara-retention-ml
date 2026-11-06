import { Link } from '@tanstack/react-router'
import { Sparkles, ShieldCheck, Zap, LineChart, Lock } from 'lucide-react'
import { SignUpForm } from './components/sign-up-form'

export function SignUp2() {
  return (
    <div className='relative min-h-screen w-full lg:grid lg:grid-cols-2'>
      {/* Left Column: Sign-up Form */}
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
            <h1 className='text-2xl font-bold tracking-tight text-foreground'>Create CRM Account</h1>
            <p className='text-xs text-muted-foreground'>
              Get immediate access to customer churn alerts, CLV forecasts, and retention playbooks.
            </p>
          </div>

          <SignUpForm />

          <div className='text-center text-xs text-muted-foreground'>
            Already have an account?{' '}
            <Link to='/sign-in-2' className='font-semibold text-indigo-600 hover:text-indigo-500 underline underline-offset-4 dark:text-indigo-400'>
              Sign In (2 Col)
            </Link>
            {' '}or{' '}
            <Link to='/sign-in' className='font-semibold text-indigo-600 hover:text-indigo-500 underline underline-offset-4 dark:text-indigo-400'>
              Simple Sign In
            </Link>
          </div>
        </div>

        {/* Footer Disclaimer */}
        <div className='text-center text-[11px] text-muted-foreground'>
          Enterprise SSO & Directory Sync Available • ISO 27001 & SOC2 Certified
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
            Enterprise Onboarding Open
          </div>
          <div className='text-xs font-mono text-slate-400'>
            UK Retail Data Core
          </div>
        </div>

        {/* Middle Value Props */}
        <div className='relative z-10 space-y-6 my-auto max-w-lg'>
          <div>
            <div className='inline-flex items-center gap-1.5 rounded-md bg-indigo-500/20 px-2.5 py-1 text-xs font-semibold text-indigo-300 border border-indigo-500/30 mb-3'>
              <Sparkles className='size-3.5' />
              Full-Stack Retention Intelligence
            </div>
            <h2 className='text-3xl font-extrabold tracking-tight leading-tight text-white'>
              Turn Raw Transactions Into Actionable Retention.
            </h2>
            <p className='mt-2 text-sm text-slate-300 leading-relaxed'>
              Seamlessly bridge transactional order streams with predictive machine learning models to prevent churn and maximize customer lifetime value.
            </p>
          </div>

          {/* Feature Highlights Grid */}
          <div className='space-y-3'>
            <div className='flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3.5 backdrop-blur-md'>
              <div className='flex size-8 shrink-0 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400'>
                <ShieldCheck className='size-4' />
              </div>
              <div className='text-xs'>
                <div className='font-semibold text-white'>Enterprise Security & RBAC</div>
                <div className='text-slate-300 mt-0.5'>Role-based access permissions, audit logs, and encrypted customer data pipelines.</div>
              </div>
            </div>

            <div className='flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3.5 backdrop-blur-md'>
              <div className='flex size-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-400'>
                <Zap className='size-4' />
              </div>
              <div className='text-xs'>
                <div className='font-semibold text-white'>Automated CS Playbooks</div>
                <div className='text-slate-300 mt-0.5'>Instant trigger webhooks for high-risk accounts to schedule executive outreach.</div>
              </div>
            </div>

            <div className='flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 p-3.5 backdrop-blur-md'>
              <div className='flex size-8 shrink-0 items-center justify-center rounded-lg bg-amber-500/20 text-amber-400'>
                <LineChart className='size-4' />
              </div>
              <div className='text-xs'>
                <div className='font-semibold text-white'>Live Portfolio Health Tracking</div>
                <div className='text-slate-300 mt-0.5'>Real-time tracking across Champions, Loyal Customers, and At-Risk customer tiers.</div>
              </div>
            </div>
          </div>

          {/* Trust Banner */}
          <div className='rounded-xl border border-white/10 bg-gradient-to-r from-white/10 to-white/5 p-4 backdrop-blur-md'>
            <div className='flex items-center gap-2 mb-1.5 text-xs font-semibold text-white'>
              <Lock className='size-3.5 text-indigo-400' />
              Built for High-Velocity Retail Teams
            </div>
            <p className='text-xs text-slate-300 leading-relaxed'>
              Integrated with Vantara&apos;s XGBoost Churn Predictor (80.8% ROC-AUC) and Ridge CLV Forecaster (R² 0.845) across 5,225 active enterprise accounts.
            </p>
          </div>
        </div>

        {/* Bottom Metrics Bar */}
        <div className='relative z-10 grid grid-cols-3 gap-4 border-t border-white/10 pt-4 text-center'>
          <div>
            <div className='text-lg font-bold text-white font-mono'>5,225</div>
            <div className='text-[10px] text-slate-400 uppercase tracking-wider'>Customer Base</div>
          </div>
          <div>
            <div className='text-lg font-bold text-emerald-400 font-mono'>&pound;3.3M+</div>
            <div className='text-[10px] text-slate-400 uppercase tracking-wider'>Protected Revenue</div>
          </div>
          <div>
            <div className='text-lg font-bold text-indigo-400 font-mono'>99.9%</div>
            <div className='text-[10px] text-slate-400 uppercase tracking-wider'>System Uptime</div>
          </div>
        </div>
      </div>
    </div>
  )
}
