export type Image = {
    src: string;
    alt?: string;
    caption?: string;
};

export type Link = {
    text: string;
    href: string;
};

export type Hero = {
    title?: string;
    text?: string;
    image?: Image;
    actions?: Link[];
};

export type Subscribe = {
    title?: string;
    text?: string;
    formUrl: string;
};

export type SiteConfig = {
    logo?: Image;
    title: string;
    subtitle?: string;
    description: string;
    image?: Image;
    headerNavLinks?: Link[];
    footerNavLinks?: Link[];
    socialLinks?: Link[];
    hero?: Hero;
    subscribe?: Subscribe;
    postsPerPage?: number;
    projectsPerPage?: number;
};

const siteConfig: SiteConfig = {
    title: 'Jacques',
    subtitle: 'ML & Data Engineer',
    description: 'Jacques Dumora Portfolio , Ml & Data enginneer',
    headerNavLinks: [
        {
            text: 'Home',
            href: '/'
        },
        {
            text: 'Projects',
            href: '/projects'
        },
        {
            text: 'Blog',
            href: '/blog'
        },
        {
            text: 'Tags',
            href: '/tags'
        }
    ],
    footerNavLinks: [
        {
            text: 'About',
            href: '/about'
        },
        {
            text: 'Contact',
            href: '/contact'
        }
    ],
    socialLinks: [
        {
            text: 'Github',
            href: 'https://github.com/Jacqkues'
        },
        {
            text: 'LinkedIn',
            href: 'www.linkedin.com/in/jacques-dumora'
        },
        {
            text: 'X/Twitter',
            href: 'https://x.com/_jacqD'
        }
    ],
    hero: {
        title: "Hi 👋",
      /*  image:{
            src:"./nasa.jpg"
        },*/
        text:"I'm **Jacques Dumora**, a ML & Data Engineer based in France. Passionate about cutting-edge technologies and data science, I specialize in crafting innovative and efficient solutions to tackle both today's challenges and those of the future.\n\n Whether it's **machine learning**, **data modeling**, or **AI-driven systems**, my goal is to merge **technical excellence** with **elegant design** to provide real value to my clients. \n \n Feel free to explore some of my coding endeavors on <a href='https://github.com/Jacqkues'>GitHub</a> or follow me on <a href='https://x.com/_jacqD'>Twitter/X</a>.",
        //text: "I'm **Jacques Dumora**, a freelance AI & Data Engineer in France. Passionate about cutting-edge technologies and data analysis, I'm here to help you find effective, aesthetic and innovative solutions to solve today's problems and anticipate tomorrow's needs.. Feel free to explore some of my coding endeavors on <a href='https://github.com/JustGoodUI/dante-astro-theme'>GitHub</a> or follow me on <a href='https://twitter.com/justgoodui'>Twitter/X</a>.",
        actions: [
            {
                text: 'Get in Touch',
                href: '/contact'
            }
        ]
    },
    postsPerPage: 8,
    projectsPerPage: 8
};

export default siteConfig;
