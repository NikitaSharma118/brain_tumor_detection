import styles from '../style';
import { Homepage } from '../assets';
import Button from './Button';

const Hero = () => (
    <section id="home" className={`flex md:flex-row flex-col ${styles.paddingY} gap-16`}>
        <div className={`flex-1 ${styles.flexStart} flex-col xl:px-0 sm:px-16 px-6`}>

          <div className='flex flex-row justify-between items-center w-full'>
              <h1 className='flex-1 font-poppins font-semibold 
              ss:text-[128px] text-[70px] text-white ss:leading-[100px] leading-[75px] mt-10'>
                  Early Detection. <br className='sm:block hidden' /> {" "}
                 <span className="text-gradient">Confident Decisions  </span> {" "}
              </h1>
          </div>

          <p className={`${styles.paragraph} max-w-[470px] mt-5`}>Transform your MRI scans into precise, actionable insights using our 
          Deep Learning platform. Revolutionizing brain tumor detection with AI-powered accuracy and efficiency. Get rapid results without wait because every second matters.</p>
            <Button styles="ml-0 mt-10"/>
        </div>

        <div >
            <img src={Homepage} alt="homepage" className="w-[100%] h-[95%] ml-8 mt-15 relative z-[5]"/>
        </div>
    </section>
)

export default Hero;